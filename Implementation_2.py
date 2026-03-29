import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from scipy.optimize import curve_fit

import pycwt as wavelet

# --- Force Publication-Quality Fonts (Times New Roman / Serif) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] =['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 1.0

# ==========================================
# Generic Noise Model Function
# ==========================================
def generic_noise_model(nu, A, s, C):
    return A * (nu**s) + C

# --- Step 1: Data Loading ---
folder = r"C:\Users\HP\OneDrive\Desktop\DeepLearning\FFT\aia_sdo-1700\*.fits"
files = sorted(glob.glob(folder))
nt = len(files)

m0 = sunpy.map.Map(files[0])

bottom_left = SkyCoord(-10*u.arcsec, -10*u.arcsec, frame=m0.coordinate_frame)
top_right = SkyCoord(10*u.arcsec, 10*u.arcsec, frame=m0.coordinate_frame)

print("Loading data for wavelet analysis...")
sizes =[sunpy.map.Map(f).submap(bottom_left=bottom_left, top_right=top_right).data.shape for f in files]
ny, nx = min(s[0] for s in sizes), min(s[1] for s in sizes)
cube = np.zeros((ny, nx, nt), dtype=np.float32)
ref = m0.submap(bottom_left=bottom_left, top_right=top_right).data[:ny, :nx]

for i, f in enumerate(files):
    img = sunpy.map.Map(f).submap(bottom_left=bottom_left, top_right=top_right).data[:ny, :nx]
    shift_est, _, _ = phase_cross_correlation(ref, img)
    cube[:, :, i] = shift(img, shift_est)

# --- Step 2: Extract & Standardize EXACTLY like Panel (a) ---
ts_raw = cube[ny//2, nx//2, :]
ts = (ts_raw - np.mean(ts_raw)) / np.std(ts_raw)

dt = 24.0 / 60.0  
time = np.arange(nt) * dt

# --- Step 3: Wavelet Transform ---
print("Computing Morlet Wavelet Transform...")
dj = 1/12           
s0 = 2 * dt         
J = 7 / dj          

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(ts, dt, dj, s0, J, wavelet='morlet')
power = np.abs(wave) ** 2
period = 1.0 / freqs

# --- FIX: Safe Normalization ---
# We only find the max power for periods <= 32 to ignore the massive bottom-edge artifact!
valid_plot_mask = period <= 32.0 
norm_power = power / np.max(power[valid_plot_mask, :]) * 0.50

# --- Step 4: 95% Confidence Levels ---
signif, _ = wavelet.significance(1.0, dt, scales, 0, alpha=0.05, significance_level=0.95, wavelet='morlet')
sig95_2d = power / signif[:, None]

# --- Step 5: EXACT REPLICA PLOTTING ---
print("Generating exact replica figure...")
fig = plt.figure(figsize=(7, 8))

# ------------- PANEL (a) -------------
ax1 = plt.subplot(2, 1, 1)

ax1.plot(time, ts, color='black', linewidth=0.6)
ax1.axhline(0, color='red', linestyle='--', linewidth=1.2, dashes=(5, 5)) 

ax1.set_xlim(0, time.max())
ax1.set_ylim(-3.5, 3.5)

ax1.set_ylabel('Intensity', fontsize=14, fontweight='bold', fontstyle='italic')
ax1.set_xlabel('Time (Minutes)', fontsize=14, fontweight='bold', fontstyle='italic')

ax1.tick_params(which='major', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
ax1.tick_params(which='minor', direction='in', length=3, width=0.5, top=True, right=True)
ax1.minorticks_on()

ax1.text(0.02, 0.88, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', fontstyle='italic')

# ------------- PANEL (b) -------------
ax2 = plt.subplot(2, 1, 2)

levels = np.linspace(0, 0.50, 100)
cf = ax2.contourf(time, np.log2(period), norm_power, levels=levels, cmap='nipy_spectral', extend='max')

ax2.contour(time, np.log2(period), sig95_2d, [-99, 1], colors='white', linewidths=1.5)

ax2.fill_between(time, np.log2(coi), np.log2(period[-1]), facecolor='black', alpha=1.0)
ax2.fill_between(time, np.log2(coi), np.log2(period[-1]), facecolor='none', edgecolor='white', hatch='xx', linewidth=0.5)

ax2.set_xlim(0, time.max())
# --- FIX: Strictly cut off the Y-axis at exactly 32 so the artifact is completely hidden ---
ax2.set_ylim(np.log2(32), np.log2(1)) 

ax2.set_ylabel('Period (Minutes)', fontsize=14, fontweight='bold', fontstyle='italic')
ax2.set_xlabel('Time (Minutes)', fontsize=14, fontweight='bold', fontstyle='italic')

y_ticks =[1, 2, 4, 8, 16, 32]
ax2.set_yticks(np.log2(y_ticks))
ax2.set_yticklabels(y_ticks, fontsize=12, fontweight='bold')

ax2.tick_params(which='major', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
ax2.tick_params(which='minor', direction='in', length=3, width=0.5, top=True, right=True)
ax2.minorticks_on()

ax2.text(0.02, 0.88, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', fontstyle='italic', color='white')

cbar = fig.colorbar(cf, ax=ax2, pad=0.02)
cbar.set_label('Wavelet Power', fontsize=14, fontweight='bold', fontstyle='italic')
cbar.set_ticks([0.00, 0.12, 0.25, 0.38, 0.50])
cbar.ax.set_yticklabels(['0.00', '0.12', '0.25', '0.38', '0.50'], fontweight='bold', fontstyle='italic', fontsize=12)
cbar.ax.tick_params(direction='in', length=4)

plt.tight_layout()
plt.savefig("Exact_Replica_Fig4_AIA1700_Final.png", dpi=400, bbox_inches='tight')
plt.show()
