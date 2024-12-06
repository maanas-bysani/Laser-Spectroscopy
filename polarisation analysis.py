import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.constants as const
from scipy.optimize import curve_fit

def sinusoidal_function(x, amp = 1, freq = 1, phase = 0, offset = 0):
    return amp * np.sin(2 * np.pi * freq * x + phase) + offset

def analysis(all_plots = False):
    data = pd.read_csv(r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 6 dec\linear polariser data.csv")
    print(data)
    angles = data.iloc[:,0]
    means = data.iloc[:,1]
    maxs = data.iloc[:,2]
    std_devs = data.iloc[:,3]/1000
    samples = data.iloc[:,4]

    if all_plots is True:
        # amp, freq, phase, offset
        p0 = [300, 0.005, -95, 450]
        popt, cov = curve_fit(sinusoidal_function, angles, means, p0)
        plt.scatter(angles, means, marker = 'x', label = 'Data')
        # plt.plot(angles, sinusoidal_function(angles, *p0), label = 'Guess')
        plt.plot(angles, sinusoidal_function(angles, *popt), label = 'Fit', color = 'black')
        plt.xlabel('Angle (°)')
        plt.ylabel('Power (\u03bcW)')
        plt.title('Linear Polariser Plot')
        plt.legend()
        plt.show()

    # normalised
    means_scaled = (means - np.min(means)) / (np.max(means) - np.min(means))
    std_devs_scaled = std_devs / (np.max(means) - np.min(means))  # Scale the errors similarly

    p0 = [0.5, 0.005, -95, 0.5]
    popt, cov = curve_fit(sinusoidal_function, angles, means_scaled, p0)
    y_fit = sinusoidal_function(angles, *popt)

    start_index = 0
    midpoint = (len(means_scaled) // 2) - 2
    data_or_fit = y_fit

    min1 = start_index + np.argmin(data_or_fit[start_index:midpoint])
    min2 = midpoint + np.argmin(data_or_fit[midpoint:])
    max1 = start_index + np.argmax(data_or_fit[start_index:midpoint])
    max2 = midpoint + np.argmax(data_or_fit[midpoint:])

    plt.errorbar(angles, means_scaled, xerr = 1, yerr = std_devs_scaled,\
                capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
                color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')
    # plt.plot(angles, sinusoidal_function(angles, *p0), label = 'Guess')
    plt.plot(angles, y_fit, label = 'Fit', color = 'black')
    plt.axvline(angles[min1], color='blue', linestyle='--', label='Min 1')
    plt.axvline(angles[min2], color='red', linestyle='--', label='Min 2')
    plt.axvline(angles[max1], color='green', linestyle='--', label='Max 1')
    plt.axvline(angles[max2], color='orange', linestyle='--', label='Max 2')
    plt.xlabel('Angle (°)')
    plt.ylabel('Scaled Power (arb. units)')
    plt.title('Linear Polariser Plot')
    plt.legend()
    plt.show()


analysis()