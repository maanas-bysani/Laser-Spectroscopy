
# import scipy.constants as const

# h = const.h
# c = const.c
# e = const.e
# f = 384.2304844685e12
# energy = h * f
# print(energy/e)

# print(c)
# lambdaa = c / f
# print(lambdaa)



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Load the data from the uploaded file
file_path = 'subtracted_spectrum.txt'
data = np.loadtxt(file_path)

# Define x-axis values (assuming equally spaced points)
x = np.arange(len(data))

# Identify peaks in the data
peaks, _ = find_peaks(data, height=0)  # Adjust threshold as needed

# Define Gaussian function
def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu)**2) / (2 * sigma**2)) + offset

# Fit Gaussian to each peak
params_list = []  # To store parameters for each fitted peak

for peak in peaks:
    # Define fitting range around the peak
    fit_range = 10  # Number of points on either side of the peak
    start = max(peak - fit_range, 0)
    end = min(peak + fit_range, len(data) - 1)
    
    x_fit = x[start:end]
    y_fit = data[start:end]

    # Initial guess for Gaussian parameters: [amp, mu, sigma, offset]
    initial_guess = [data[peak], peak, .03, np.min(data)]

    try:
        # Fit Gaussian
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=initial_guess, maxfev = 1000000000)
        params_list.append(popt)
    except RuntimeError:
        # Skip if the fit fails
        params_list.append(None)

# Plot the original data and fitted Gaussians
plt.figure(figsize=(10, 6))
plt.plot(x, data, label='Original Data')
for params in params_list:
    if params is not None:
        amp, mu, sigma, offset = params
        plt.plot(x, gaussian(x, *params), label=f'Fit at mu={mu:.2f}')

plt.legend()
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Gaussian Fits to Peaks')
plt.show()
