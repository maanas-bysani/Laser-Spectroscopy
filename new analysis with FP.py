import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.constants as const
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def gaussian(x, amp, mu, sigma, c):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c

def n_gaussians(x, *params):
    if len(params) % 4 != 0:
        raise ValueError("Number of parameters must be a multiple of 4 (amp, mu, sigma, c for each Gaussian).")
    
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        amp, mu, sigma, c = params[i:i+4]
        result += amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c    
    return result

def lorentzian_with_offset(x, amp, mean, width, offset):
    return amp / (1 + ((x - mean) / width) ** 2) + offset

def std_dev_to_fwhm(std_devs):
    conversion_factor = 2 * np.sqrt(2 * np.log(2))
    fwhms = conversion_factor * np.array(std_devs)
    return fwhms

def analysis(hfs_file, fs_file, fp_file, peak_height_guess, peak_separation_guess, window_size_input, all_plots = False):
    hfs_data = pd.read_csv(hfs_file)
    fs_data = pd.read_csv(fs_file)
    fp_data = pd.read_csv(fp_file)
    # all_data = pd.read_csv(r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 5 dec\ALL.csv") 

    index = hfs_data.index
    time = hfs_data.iloc[:,0] - hfs_data.iloc[:,0][0]
    time_fp = fp_data.iloc[:,0] - fp_data.iloc[:,0][0]

    # channel_1 = hfs_data.iloc[:,1]
    # channel_2 = hfs_data.iloc[:,2]
    # channel_3 = hfs_data.iloc[:,3]
    ttl_signal = hfs_data.iloc[:,-1]
    ttl_signal_fp = fp_data["C4 in V"]

    # if all_plots is True:
        # plt.plot(time, channel_1, label = 'Channel 1')
        # plt.plot(time, channel_2, label = 'Channel 2')
        # plt.plot(time, channel_3, label = 'Channel 3')
        # plt.plot(time, ttl_signal, label = 'TTL Signal')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Voltage (V)')
        # plt.legend()
        # plt.show()

    # actual needed data
    hfs_channel = hfs_data["C2 in V"]
    fs_channel = fs_data["C2 in V"]
    fp_channel = fp_data["C3 in V"]
    ttl_signal = hfs_data["C4 in V"]

    plt.plot(time, fs_channel, label = 'FS')
    plt.plot(time, hfs_channel, label = 'HFS')
    # plt.plot(time, fp_channel, label = 'FP')
    plt.plot(time, ttl_signal, label = 'TTL Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()


    if all_plots is True:
        plt.subplot(3, 1, 1)
        plt.plot(time, ttl_signal/(5), label = 'Scaled TTL Signal', lw = 0.5, alpha = 0.5)
        plt.plot(time, fs_channel, label = 'Fine Structure')
        # plt.title("Fine Structure")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time, ttl_signal/(5), label = 'Scaled TTL Signal', lw = 0.5, alpha = 0.5)
        plt.plot(time, hfs_channel, label = 'Hyperfine Structure')
        # plt.title("Hyperfine Structure")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(time_fp, ttl_signal_fp/(1000), label = 'Scaled TTL Signal', lw = 0.5, alpha = 0.5)
        plt.plot(time_fp, fp_channel, label = 'Fabry Perot Signal')
        # plt.title("Fabry Perot Signal")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.show()


    # subtracted spectrum
    subtracted_spectrum = hfs_channel - fs_channel
    plt.plot(time, subtracted_spectrum, label = "HFS for Rb 85 and 87", lw = 0.75)
    plt.title("Subtracted Spectrum")
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()

    # fabry perot 



    peaks, properties = find_peaks(fp_channel, height = peak_height_guess, distance = peak_separation_guess)  # distance is index number (not x axis value)
    peaks_indices = peaks
    peaks_heights = properties['peak_heights']

    if all_plots is True:
        plt.plot(fp_channel, label='Fabry Perot Signal')
        plt.scatter(peaks_indices, peaks_heights, color='red', label='Maxima Peaks', zorder=5, marker = 'x')
        plt.title('Fabry Perot Peaks')
        plt.xlabel('Index')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.show()

    if all_plots is True:
        plt.plot(fp_channel, label='Fabry Perot Signal')
        plt.scatter(peaks_indices, peaks_heights, color='red', label='Maxima Peaks', zorder=5, marker = 'x')
        plt.xticks(peaks_indices, range(1, len(peaks_indices) + 1))
        plt.title('Fabry Perot Peaks')
        plt.xlabel('Peak Index')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.show()

    # amp, mean, width, offset
    amp_list = []
    mean_list = []
    width_list = []
    offset_list = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    
    for idx, peak_idx in enumerate(peaks_indices):
        window_size = window_size_input
        start = max(peak_idx - window_size, 0)
        end = min(peak_idx + window_size, len(fp_channel))
        region_x = np.arange(start, end)
        region_y = fp_channel[start:end]

        # amp, mean, width, offset
        initial_guess = [peaks_heights[peaks_indices.tolist().index(peak_idx)], peak_idx, 5, np.min(region_y)]

        popt, cov = curve_fit(lorentzian_with_offset, region_x, region_y, p0=initial_guess)
        fitted_curve = lorentzian_with_offset(region_x, *popt)

        amp_list.append(popt[0])
        mean_list.append(popt[1])
        width_list.append(popt[2])
        offset_list.append(popt[3])

        if idx == 0:
            ax2.plot(region_x, fitted_curve, color='green', linestyle='--', label='Lorentzian Fit')
            # plt.plot(region_x, fitted_curve, color='green', linestyle='--')
        else:
            ax2.plot(region_x, fitted_curve, color='green', linestyle='--')
            # plt.plot(region_x, fitted_curve, color='green', linestyle='--')

    # plt.plot(time_fp, fp_channel, label='Signal', alpha = 0.4, lw = 0.7)
    # plt.scatter(peaks_indices, peaks_heights, color='red', label='Maxima Peaks', zorder=5, marker='x')
    # plt.title('Signal with Lorentzian Fits')
    # plt.xlabel('Peak Index')
    # plt.ylabel('Voltage (V)')
    # plt.xticks(peaks_indices, range(1, len(peaks_indices) + 1))
    # plt.legend()
    # plt.show()

    ax1.plot(time_fp*1000, fp_channel, label='Signal', alpha = 0.4, lw = 0.7)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)')

    ax2.scatter(peaks_indices, peaks_heights, color='red', label='Lorentzian Peaks', zorder=5, marker='x', s = 20, linewidths=0.5)
    ax2.set_xlabel('Peak Index')
    ax2.set_xticks(peaks_indices, range(1, len(peaks_indices) + 1))    

    plt.title('Signal with Lorentzian Fits')
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    plt.legend(handles=handles, labels=labels, loc='best')
    plt.show()



    if all_plots is True:
        mean_array = np.array(mean_list) # in index numbers
        mean_diffs = np.diff(mean_array, axis=0)
        plt.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs, marker = 'x')
        plt.xlabel('Peak Index')
        plt.ylabel('Difference in Mean of Lorentzian Fit (index number)')
        plt.title('Index Number Difference vs Peak Number')
        plt.show()

        mean_array = np.array(mean_list, dtype = int)
        mean_times = time_fp[mean_array]
        mean_diffs = np.diff(mean_times, axis=0)
        plt.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs, marker = 'x')
        plt.xlabel('Peak Index')
        plt.ylabel('Difference in Mean of Lorentzian Fit (time)')
        plt.title('Time Scale Difference vs Peak Number')
        plt.show()


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    mean_array = np.array(mean_list) # in index numbers
    mean_diffs = np.diff(mean_array, axis=0)
    ax2.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs, marker = 'x')
    ax2.set_xlabel('Peak Index')
    ax2.set_ylabel('Difference in Mean of Lorentzian Fit (index number)')
    # plt.title('Index Number Difference vs Peak Number')
    # plt.show()

    mean_array = np.array(mean_list, dtype = int) # in index numbers
    mean_times = time_fp[mean_array]*1000
    mean_diffs = np.diff(mean_times, axis=0)
    ax1.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs, marker = 'x')
    ax1.set_xlabel('Peak Index')
    ax1.set_ylabel('Difference in Mean of Lorentzian Fit (time (ms))')

    plt.title('Variation in Difference between Peak Means')
    plt.show()


peak_height_guess_dec_5 = 0.007
peak_separation_guess_dec_5 = 5000
window_size_input_dec_5 = 2000

hfs_file_dec_5 = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 5 dec\HFS1.csv"
fs_file_dec_5 = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 5 dec\FS1.csv"
fp_file_dec_5 = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 5 dec\FP1.csv"
dec_5_data_params = [hfs_file_dec_5, fs_file_dec_5, fp_file_dec_5, peak_height_guess_dec_5, peak_separation_guess_dec_5, window_size_input_dec_5]


peak_height_guess_dec_6 = 0.006
peak_separation_guess_dec_6 = 500
window_size_input_dec_6 = 2000

hfs_file_dec_6 = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 6 dec\HFS1.csv"
fs_file_dec_6 = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 6 dec\FS1.csv"
fp_file_dec_6 = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 6 dec\FP1.csv"

dec_6_data_params = [hfs_file_dec_6, fs_file_dec_6, fp_file_dec_6, peak_height_guess_dec_6, peak_separation_guess_dec_6, window_size_input_dec_6]

params = dec_6_data_params

analysis(*params, all_plots=False)

# dec_6_data_params = [hfs_file, fs_file, fp_file, peak_height_guess, peak_separation_guess, window_size_input]
