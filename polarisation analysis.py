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

    # plt.errorbar(angles, means_scaled, xerr = 1, yerr = std_devs_scaled,\
    #             capsize = 2, elinewidth = 1, capthick = 1, barsabove = False, fmt = 'x', \
    #             color = 'black', alpha = 1, ecolor = 'tab:blue', label='Data')
    # # plt.plot(angles, sinusoidal_function(angles, *p0), label = 'Guess')
    # plt.plot(angles, y_fit, label = 'Fit', color = 'black')
    # plt.axvline(angles[min1], color='blue', linestyle='--', label='Min 1')
    # plt.axvline(angles[min2], color='red', linestyle='--', label='Min 2')
    # plt.axvline(angles[max1], color='green', linestyle='--', label='Max 1')
    # plt.axvline(angles[max2], color='orange', linestyle='--', label='Max 2')
    # plt.xlabel('Angle (°)')
    # plt.ylabel('Scaled Power (arb. units)')
    # plt.title('Linear Polariser Plot')
    # plt.legend()
    # plt.show()


    # plt.figure(figsize=(5,5))
    # # # plt.polar(angles * np.pi / 180, means/np.max(means), 'x', label = 'Data')
    # # plt.polar(angles * np.pi / 180, means/np.max(means), label = 'Data')
    # # plt.title("Polarisation of Input Beam", pad = 25)
    # # plt.ylabel('Normalised Power', rotation=45, size=10)


    # ax = plt.subplot(112, polar=True)
    # c = plt.plot(angles * np.pi / 180, means/np.max(means),label='Data')
    # ax.set_title("Polarisation of Input Beam", pad = 25)
    # ax.set_ylabel('Normalised Power', rotation=21, size=10, labelpad=20)
    # ax.yaxis.set_label_coords(.83, 0.56)  # (x, y) coordinates in axes fraction

    # plt.tight_layout()
    # # plt.savefig('polarisation.png', dpi=1200)
    # plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={})

    # Linear Plot (Left)
    axes[0].errorbar(angles, means_scaled, xerr=1, yerr=std_devs_scaled,
                    capsize=2, elinewidth=1, capthick=1, barsabove=False, fmt='x',
                    color='black', alpha=1, ecolor='tab:blue', label='Data')
    axes[0].plot(angles, y_fit, label='Fit', color='black')
    axes[0].axvline(angles[min1], color='blue', linestyle='--', label='Min 1')
    axes[0].axvline(angles[min2], color='red', linestyle='--', label='Min 2')
    axes[0].axvline(angles[max1], color='green', linestyle='--', label='Max 1')
    axes[0].axvline(angles[max2], color='orange', linestyle='--', label='Max 2')
    axes[0].set_xlabel('Angle (°)')
    axes[0].set_ylabel('Scaled Power (arb. units)')
    axes[0].set_title('Linear Polariser Plot')
    axes[0].legend()

    # Polar Plot (Right)
    ax = plt.subplot(122, polar=True)
    c = plt.plot(angles * np.pi / 180, means/np.max(means),label='Data')
    ax.set_title("Polarisation of Input Beam", pad = 25)
    ax.set_ylabel('Normalised Power', rotation=21, size=10, labelpad=20)
    ax.yaxis.set_label_coords(.83, 0.56)  # (x, y) coordinates in axes fraction


    # polar_ax = fig.add_subplot(1, 2, 2, polar=True)
    # polar_ax.plot(angles * np.pi / 180, means / np.max(means), label='Data')
    # polar_ax.set_title("Polarisation of Input Beam", pad=22)
    # polar_ax.set_ylabel('Normalised Power', rotation=19, size=10, labelpad=20)
    # polar_ax.yaxis.set_label_coords(0.8, 0.57)
    # polar_ax.spines['polar'].set_visible(False)


    # Adjust layout and show
    plt.tight_layout()
    plt.show()


    means_scaled = means / np.max(means)
    std_dev_rel_error = std_devs / means
    std_devs_scaled = means_scaled * std_dev_rel_error

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


    plt.figure(figsize=(10, 5))

    # Linear Plot (Left)
    ax1 = plt.subplot(1, 2, 1)  # Create the first subplot
    ax1.errorbar(angles, means_scaled, xerr=1, yerr=std_devs_scaled,
                capsize=2, elinewidth=1, capthick=1, barsabove=False, fmt='x',
                color='black', alpha=1, ecolor='tab:blue', label='Data')
    ax1.plot(angles, y_fit, label='Fit', color='black')
    ax1.axvline(angles[min1], color='blue', linestyle='--', label='Min 1')
    ax1.axvline(angles[min2], color='red', linestyle='--', label='Min 2')
    ax1.axvline(angles[max1], color='green', linestyle='--', label='Max 1')
    ax1.axvline(angles[max2], color='orange', linestyle='--', label='Max 2')
    ax1.set_xlabel('Angle (°)')
    ax1.set_ylabel('Normalised Power (arb. units)') #(μW)
    ax1.set_title('Linear Polariser Plot')
    ax1.legend()

    # Polar Plot (Right)
    ax2 = plt.subplot(1, 2, 2, polar=True)  # Create the second subplot (polar)
    ax2.plot(angles * np.pi / 180, means / np.max(means), label='Data')
    ax2.set_title("Polarisation of Input Beam", pad=22)
    ax2.set_ylabel('Normalised Power (arb. units)', rotation=19, size=10, labelpad=20)
    ax2.yaxis.set_label_coords(0.78, 0.55)

    # Remove the polar plot border
    # ax2.spines['polar'].set_visible(False)

    # Remove angular (x-axis) ticks, keep radial (y-axis) ticks
    # ax2.set_xticks([])  # Remove angular ticks only

    # Adjust layout and show
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust space to leave room for titles
    plt.savefig('polarisation.png', dpi=1200)
    plt.show()

analysis()