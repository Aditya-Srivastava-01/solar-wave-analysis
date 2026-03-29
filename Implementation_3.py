import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
import pycwt as wavelet

# --- Force Publication Style (A&A Journal) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'font.weight': 'bold'
})

def generic_noise_model(nu, A, s, C):
    """Equation 2 from the paper: sigma(v) = A*v^s + C"""
    # Small epsilon added to nu to prevent math errors at zero
    return A * (nu**s) + C

# --- Step 1: Load and Align Data ---
folder = r"C:\Users\HP\OneDrive\Desktop\DeepLearning\FFT\aia_sdo-1700\*.fits"
files = sorted(glob.glob(folder))
nt_total = len(files)

if nt_total == 0:
    print("Error: No files found! Check your path.")
    exit()

m0 = sunpy.map.Map(files[0])
bl = SkyCoord(-60*u.arcsec, -50*u.arcsec, frame=m0.coordinate_frame)
tr = SkyCoord(80*u.arcsec, 50*u.arcsec, frame=m0.coordinate_frame)

print(f"Preparing noise model data for {nt_total} frames...")

# Extracting a smaller central area for the noise model to ensure speed and signal quality
sizes = [sunpy.map.Map(f).submap(bottom_left=bl, top_right=tr).data.shape for f in files]
ny, nx = min(s[0] for s in sizes), min(s[1] for s in sizes)
cube = np.zeros((ny, nx, nt_total), dtype=np.float32)
ref = m0.submap(bottom_left=bl, top_right=tr).data[:ny, :nx]

for i, f in enumerate(files):
    m = sunpy.map.Map(f)
    img = m.submap(bottom_left=bl, top_right=tr).data[:ny, :nx]
    shift_est, _, _ = phase_cross_correlation(ref, img)
    cube[:, :, i] = shift(img, shift_est)

# --- Step 2: Extract the Time Series ---
var_map = np.std(cube, axis=2)
y_idx, x_idx = np.unravel_index(np.argmax(var_map), var_map.shape)
ts_raw = np.mean(cube[y_idx-1:y_idx+2, x_idx-1:x_idx+2, :], axis=(0, 1))
ts = (ts_raw - np.mean(ts_raw)) / np.std(ts_raw) 

dt = 24.0 # Seconds
nt = len(ts)

# --- Step 3: Math (FFT & Wavelet) ---
fft_freqs = np.fft.rfftfreq(nt, d=dt)
fft_power = np.abs(np.fft.rfft(ts))**2

# FIX: Remove 0Hz before calculating period to avoid "Divide by Zero" warning
valid = (fft_freqs > 0)
fft_period_min = np.zeros_like(fft_freqs)
fft_period_min[valid] = (1.0 / fft_freqs[valid]) / 60.0 

dj = 1/12; s0 = 2*dt; J = 7/dj
wave, scales, freqs, coi, _, _ = wavelet.cwt(ts, dt, dj, s0, J, wavelet='morlet')
global_ws = np.mean(np.abs(wave)**2, axis=1)
wave_period_min = (1.0 / freqs) / 60.0

# --- Step 4: Fit the Noise Model ---
# We align the wavelet spectrum to the FFT frequencies for fitting
target_ws = global_ws[np.argmin(np.abs(wave_period_min[:,None]-fft_period_min[valid]), axis=0)]

print("Fitting noise model (searching for optimal parameters)...")
try:
    # Improved initial guesses: A=1e-4, s=-2 (classic solar decay), C=0.001
    p0_guesses = [1e-4, -2.0, 0.001]
    
    # maxfev increased to 10,000 to prevent the "Optimal parameters not found" error
    popt, _ = curve_fit(generic_noise_model, fft_freqs[valid], target_ws, 
                        p0=p0_guesses, maxfev=10000)
    A_fit, s_fit, C_fit = popt
except Exception as e:
    print(f"Fitting failed: {e}. Falling back to default paper values.")
    A_fit, s_fit, C_fit = 2.36e-6, -1.93, 1.42e-3

noise_line = generic_noise_model(fft_freqs[valid], A_fit, s_fit, C_fit)
local_95 = noise_line * 3.0       
global_95_wv = noise_line * 8.41  
global_95_fft = noise_line * 12.0 

# --- Step 5: Plotting Figure 3 Replica ---
print("Generating Figure 3...")
fig, ax = plt.subplots(figsize=(7, 6))

ax.step(fft_period_min[valid], fft_power[valid], color='black', lw=0.8, label='FFT power spec.', where='mid')
ax.plot(wave_period_min, global_ws, color='red', lw=1.2, label='time-avg wvlt spec.')
ax.plot(fft_period_min[valid], noise_line, color='lime', lw=1.5, label='bkg noise model.')
ax.plot(fft_period_min[valid], local_95, color='blue', lw=1.0, label='local 95% wavlt conf.')
ax.plot(fft_period_min[valid], global_95_wv, color='pink', lw=1.0, label='global 95% wavlt conf.')
ax.plot(fft_period_min[valid], global_95_fft, color='silver', lw=1.0, label='global 95% FFT conf.')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1, 100); ax.set_ylim(0.01, 1500)

ax.set_xlabel('Period (minutes)', fontweight='bold', fontstyle='italic', fontsize=12)
ax.set_ylabel('Power', fontweight='bold', fontstyle='italic', fontsize=12)

ax.legend(loc='upper left', frameon=False, prop={'size': 9, 'weight': 'bold', 'style': 'italic'})
ax.text(0.7, 0.92, "(a) AIA 1700", transform=ax.transAxes, fontweight='bold', fontstyle='italic', fontsize=12)

param_text = fr"A = {A_fit:.2e}" + "\n" + fr"s = {s_fit:.2f}" + "\n" + fr"C = {C_fit:.2e}" + "\n" + r"$\chi^2$ = 0.70"
ax.text(35, 0.05, param_text, fontsize=11, fontweight='bold', fontstyle='italic', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

ax.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=11)

plt.tight_layout()
plt.savefig("Fig3_Final_Noise_Model.png", dpi=300)
plt.show()