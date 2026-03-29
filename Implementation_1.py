import glob
import numpy as np
import matplotlib.pyplot as plt

import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord

from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from scipy.optimize import curve_fit

# ==========================================
# PAPER EXACT MATH: Generic Noise Model
# ==========================================
def generic_noise_model(nu, A, s, C):
    """Power-law background noise model: sigma(v) = A*v^s + C"""
    return A * (nu**s) + C

# --- Step 1 & 2: Data Loading and Alignment ---
# Updated with your exact folder path
folder = r"C:\Users\HP\OneDrive\Desktop\DeepLearning\FFT\aia_sdo-1700\*.fits"
files = sorted(glob.glob(folder))
nt = len(files)
print(f"Total frames: {nt}")

if nt == 0:
    raise ValueError("No FITS files found! Please check the folder path and ensure it contains .fits files.")

m0 = sunpy.map.Map(files[0])
bottom_left = SkyCoord(-70*u.arcsec, -110*u.arcsec, frame=m0.coordinate_frame)
top_right = SkyCoord(80*u.arcsec, 90*u.arcsec, frame=m0.coordinate_frame)

sizes =[]
for f in files:
    m = sunpy.map.Map(f)
    c = m.submap(bottom_left=bottom_left, top_right=top_right)
    sizes.append(c.data.shape)

ny, nx = min(s[0] for s in sizes), min(s[1] for s in sizes)
cube = np.zeros((ny, nx, nt), dtype=np.float32)

cut_ref = m0.submap(bottom_left=bottom_left, top_right=top_right)
ref = cut_ref.data[:ny, :nx]

print("Starting alignment...")
for i, f in enumerate(files):
    m = sunpy.map.Map(f)
    c = m.submap(bottom_left=bottom_left, top_right=top_right)
    img = c.data[:ny, :nx]
    shift_est, _, _ = phase_cross_correlation(ref, img)
    cube[:, :, i] = shift(img, shift_est)

# --- Step 3: Data Preprocessing ---
cube = cube[5:-5, 5:-5, :]
ny, nx, nt = cube.shape 

# Normalize by mean image (relative intensity fluctuations)
mean_img = np.mean(cube, axis=2, keepdims=True)
cube = (cube - mean_img) / mean_img

# --- Step 4: Fourier Analysis ---
print("Computing Fast Fourier Transform...")
fft_cube = np.fft.rfft(cube, axis=2)
power_cube = np.abs(fft_cube)**2

dt = 24  # seconds (AIA 1700 typical cadence)
freq = np.fft.rfftfreq(nt, dt)
freq_mhz = freq * 1000

# Remove 0 Hz (DC component) for curve fitting
valid_idx = freq > 0
freq_fit = freq[valid_idx] 
freq_mhz_fit = freq_mhz[valid_idx]
power_cube_fit = power_cube[:, :, valid_idx]

# --- Step 5: Global Noise Fitting & Normalization (THE FIX) ---
target = 5.31 # CHANGE THIS to 11.04 or 1.64 to reproduce other panels
idx = np.argmin(np.abs(freq_mhz_fit - target))
actual_target_freq = freq_mhz_fit[idx]
print(f"Selected frequency for power map: {actual_target_freq:.2f} mHz")

# 1. Average the power spectrum spatially to get a clean 1D spectrum 
mean_power_spectrum = np.mean(power_cube_fit, axis=(0, 1))

# 2. Fit the noise model ONCE to the clean average spectrum
p0_guesses =[1e-4, -1.0, 0.001] 
popt, _ = curve_fit(generic_noise_model, freq_fit, mean_power_spectrum, p0=p0_guesses, maxfev=2000)
A_fit, s_fit, C_fit = popt

# 3. Calculate expected noise exactly at the target frequency
expected_noise = generic_noise_model(freq_fit[idx], A_fit, s_fit, C_fit)

# 4. Calculate the 95% threshold scalar value (m_factor from the paper)
m_factor = 8.41  
conf_95_threshold = m_factor * expected_noise

# 5. Extract the raw 2D power map and normalize it by the threshold scalar
raw_power_map = power_cube_fit[:, :, idx]
normalized_power_map = raw_power_map / conf_95_threshold

# 6. Log scale the map (as plotted in Figure 2)
power_map_final_display = np.log10(normalized_power_map)


# --- Step 6: Visualization ---
print("Generating final figure...")
initial_cut_for_projection = m0.submap(bottom_left=bottom_left, top_right=top_right)

fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': initial_cut_for_projection})

# Dynamically set limits to match the original paper exactly based on frequency
if target < 4.0:
    vmin_val, vmax_val = -2.00, 1.00   # Paper limits for 1.64 mHz (Panel E)
elif target < 8.0:
    vmin_val, vmax_val = -1.50, 1.00   # Paper limits for 5.31 mHz (Panel C)
else:
    vmin_val, vmax_val = -2.50, 0.10   # Paper limits for 11.04 mHz (Panel A)

im = ax.imshow(power_map_final_display,
               origin="lower",
               cmap="Greens_r", 
               vmin=vmin_val,
               vmax=vmax_val) 

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Fourier Power (Log-scale)")

# Labels and correct orientation (Inverts removed to match the paper's upward/rightward progression)
ax.set_xlabel("X (arcsec)")
ax.set_ylabel("Y (arcsec)")

# Format title to match the paper style
if target < 4.0:
    panel_letter = "(E)"
elif target < 8.0:
    panel_letter = "(C)"
else:
    panel_letter = "(A)"

plt.title(f"AIA 1700: FFT\n{panel_letter} Freq: {actual_target_freq:.2f} mHz", loc='left')
plt.show()