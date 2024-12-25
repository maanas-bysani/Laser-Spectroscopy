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

def analysis(hfs_file, fs_file, fp_file, peak_height_guess, peak_separation_guess, window_size_input, \
             all_plots = False, subtracted_manual = False, subtracted_automatic = True, fabry_perot = True, fp_region_wise = True):
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

    plt.plot(time, fs_channel, label = 'FS Data')
    plt.plot(time, hfs_channel, label = 'HFS Data')
    plt.plot(time_fp, fp_channel, label = 'FP')
    plt.plot(time, ttl_signal, label = 'TTL Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Data')
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



    # curve fitting manual

    if subtracted_manual is True:
        # # amp, mu, sigma, c
        initial_guess = [0.01, 0.00078, 0.00005, -0.01] # fits
        initial_guess = [0.03, 0.0012, 0.00005, 0.0] # fits
        initial_guess = [0.02, 0.00165, 0.000025, 0.0] # fits with window size 1.4
        initial_guess = [0.15, 0.00195, 0.00005, 0.05] # fits
        initial_guess = [0.275, 0.00237, 0.00004, 0.05] # fits
        initial_guess = [0.01, 0.00313, 0.00004, 0.0] # fits
        initial_guess = [0.001, 0.0082, 0.00001, 0.05] # no work
        initial_guess = [0.03, 0.00834, 0.00002, 0.11] # fits with window 2
        initial_guess = [0.25, 0.008655, 0.00001, 0.37] # fits
        initial_guess = [0.32, 0.00882, 0.000024, 0.37] # fits with window 1.6
        initial_guess = [0.09, 0.00913, 0.00002, 0.05] # fits
        initial_guess = [0.3, 0.02194, 0.00013, 0.05] # fits
        initial_guess = [-0.022, 0.0320, 0.000036, -0.0006] # fits
        initial_guess = [0.02, 0.0322, 0.000006, 0.02] # fits with window 1.5
        initial_guess = [0.05, 0.03254, 0.00005, 0.04] # fits
        initial_guess = [0.026, 0.03285, 0.00005, 0.03] # fits

        mu_guess = initial_guess[1]
        sigma_guess = initial_guess[2]
        window_factor = 3  # number of sigma to include in the window

        time_min = mu_guess - window_factor * sigma_guess
        time_max = mu_guess + window_factor * sigma_guess

        window = (time >= time_min) & (time <= time_max)
        time_window = time[window]
        spectrum_window = subtracted_spectrum[window]

        popt, cov = curve_fit(gaussian, time_window, spectrum_window, p0=initial_guess, maxfev=1000000)
        fitted_curve_window = gaussian(time_window, *popt)
        fitted_curve_extended = gaussian(time, *popt)

        plt.plot(time, subtracted_spectrum, label = "Data", lw = 0.75)


        # plt.plot(time_window, spectrum_window, 'o', color='red', label="Fit Window Data", markersize=2)
        plt.plot(time, gaussian(time, *initial_guess), color='green', linestyle='--', label='Guess')
        plt.plot(time_window, fitted_curve_window, color='black', label='Gaussian Fit - Window')
        plt.plot(time, fitted_curve_extended, color='orange', ls = 'dashed', alpha = 0.7, label='Gaussian Fit - Extended')

        plt.title("Subtracted Spectrum")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.show()



    # curve fitting automatic

    if subtracted_automatic is True:

        initial_guesses = [
            [0.01, 0.00078, 0.00005, -0.01, 3],  # Default window = 3
            [0.03, 0.0012, 0.00005, 0.0, 3],     # Default window = 3
            [0.02, 0.00165, 0.000025, 0.0, 1.4], # Window = 1.4
            [0.15, 0.00195, 0.00005, 0.05, 3],   # Default window = 3
            [0.275, 0.00237, 0.00004, 0.05, 3],  # Default window = 3
            [0.01, 0.00313, 0.00004, 0.0, 3],    # Default window = 3
            [0.03, 0.00834, 0.00002, 0.11, 2],   # Window = 2
            [0.25, 0.008655, 0.00001, 0.37, 3],  # Default window = 3
            [0.32, 0.00882, 0.000024, 0.37, 1.6],# Window = 1.6
            [0.09, 0.00913, 0.00002, 0.05, 3],   # Default window = 3
            [0.3, 0.02194, 0.00013, 0.05, 3],    # Default window = 3
            [-0.022, 0.0320, 0.000036, -0.0006, 3], # Default window = 3
            [0.02, 0.0322, 0.000006, 0.02, 1.8], # Window = 1.8
            [0.05, 0.03254, 0.00005, 0.04, 3],   # Default window = 3
            [0.026, 0.03285, 0.00005, 0.03, 3]   # Default window = 3
        ]


        amplitudes_list = []
        means_list = []
        sigmas_list = []
        offsets_list = []
        fitted_curves_list = []
        window_times_list = []


        for initial_guess in initial_guesses:
            p0 = initial_guess[0:4]
            mu_guess = p0[1]
            sigma_guess = p0[2]
            window_factor = initial_guess[4]

            time_min = mu_guess - window_factor * sigma_guess
            time_max = mu_guess + window_factor * sigma_guess

            window = (time >= time_min) & (time <= time_max)
            time_window = time[window]
            spectrum_window = subtracted_spectrum[window]

            popt, cov = curve_fit(gaussian, time_window, spectrum_window, p0=p0, maxfev=1000000)

            amp, mu, sigma, c = popt
            amplitudes_list.append(amp)
            means_list.append(mu)
            sigmas_list.append(sigma)
            offsets_list.append(c)
            window_times_list.append(time_window)

            fitted_curves_list.append(gaussian(time_window, *popt))


        plt.plot(time, subtracted_spectrum, label='Data', alpha=0.7, lw = 0.75)

        for i, (fitted_time, fitted_curve) in enumerate(zip(window_times_list, fitted_curves_list), start=1):
            plt.plot(fitted_time, fitted_curve, label = f'G {i}')

        plt.title("Subtracted Spectrum with Gaussian Fits")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.show()


    # fabry perot 

    if fabry_perot is True:

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

        mean_array = np.array(mean_list, dtype = int) # in index numbers
        mean_diffs = np.diff(mean_array, axis=0)
        mean_times = time_fp[mean_array]*1000
        mean_diffs_time = np.diff(mean_times, axis=0)


        ax2.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs, marker = 'x')
        ax2.set_xlabel('Peak Index')
        ax2.set_ylabel('Fitted Lorentzian Spacing (index number)')

        ax1.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs_time, marker = 'x')
        ax1.set_xlabel('Peak Index')
        ax1.set_ylabel('Fitted Lorentzian Spacing (time (ms))')

        plt.title('Variation in Lorentzian Spacing')
        plt.show()


        # double x and double y axis
        if all_plots is True:
            fig=plt.figure()
            ax=fig.add_subplot(111, label="1")
            ax2=fig.add_subplot(111, label="2", frame_on=False)

            ax.scatter(np.arange(0, len(peaks_indices) - 1, 1), mean_diffs_time, color="black", marker = 'x')
            ax.set_xlabel("Peak Index")
            ax.set_ylabel("Fitted Lorentzian Spacing (time (ms))")
            ax.tick_params(axis='x')
            ax.tick_params(axis='y')

            ax2.scatter(mean_times[:-1], mean_diffs, marker = 'x', color = 'white', alpha = 0.1)
            ax2.xaxis.tick_top()
            ax2.yaxis.tick_right()
            ax2.set_xlabel('Time (ms)') 
            ax2.set_ylabel('Fitted Lorentzian Spacing (index number)')
            ax2.xaxis.set_label_position('top') 
            ax2.yaxis.set_label_position('right') 
            ax2.tick_params(axis='x' )
            ax2.tick_params(axis='y')
            plt.title('Variation in Lorentzian Spacing')
            # plt.grid()
            plt.show()


    # region wise analysis

    if fp_region_wise is True:
        r1_start = 0.000
        r1_end = 0.005
        r2_start = 0.007
        r2_end = 0.013
        r3_start = 0.019
        r3_end = 0.024
        r4_start = 0.030
        r4_end = 0.035

        # all plots overlayed
        # plt.plot(time, fs_channel/np.max(fs_channel), label = 'FS Data')
        # plt.plot(time, hfs_channel/np.max(hfs_channel), label = 'HFS Data')
        plt.plot(time, subtracted_spectrum/np.max(subtracted_spectrum), label = 'HFS', alpha = 0.5)
        plt.plot(time_fp, fp_channel/np.max(fp_channel), label = 'FP', color = 'black', alpha = 0.9, lw = 0.5)
        plt.plot(time, ttl_signal/np.max(ttl_signal), label = 'TTL Signal', alpha = 0.5)
        plt.fill_betweenx([0, 1], 0.000, 0.005, color='gray', alpha=0.25, label="R1")
        plt.fill_betweenx([0, 1], 0.007, 0.013, color='gray', alpha=0.40, label="R2")
        plt.fill_betweenx([0, 1], 0.019, 0.024, color='gray', alpha=0.55, label="R3")
        plt.fill_betweenx([0, 1], 0.030, 0.035, color='gray', alpha=0.70, label="R4")

        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Selected Regions')
        plt.legend()
        plt.show()

        # r1_time = time_fp[(time_fp >= r1_start) & (time_fp <= r1_end)]
        # r2_time = time_fp[(time_fp >= r2_start) & (time_fp <= r2_end)]
        # r3_time = time_fp[(time_fp >= r3_start) & (time_fp <= r3_end)]
        # r4_time = time_fp[(time_fp >= r4_start) & (time_fp <= r4_end)]

        # r1_signal = fp_channel[(time_fp >= r1_start) & (time_fp <= r1_end)]
        # r2_signal = fp_channel[(time_fp >= r2_start) & (time_fp <= r2_end)]
        # r4_signal = fp_channel[(time_fp >= r3_start) & (time_fp <= r3_end)]
        # r4_signal = fp_channel[(time_fp >= r4_start) & (time_fp <= r4_end)]


        regions = [(r1_start, r1_end), (r2_start, r2_end), \
                (r3_start, r3_end), (r4_start, r4_end)]

        region_peaks = {}
        index_offset = 0


        df_times = pd.DataFrame()
        df_indices = pd.DataFrame()
        df_heights = pd.DataFrame()




        for i, (start, end) in enumerate(regions, start=1):
            region_mask = (time_fp >= start) & (time_fp <= end)    
            index_offset = (time_fp - start).abs().idxmin()
            region_time = time_fp[region_mask]
            region_signal = fp_channel[region_mask]

            peaks, properties = find_peaks(region_signal, height=peak_height_guess, distance=peak_separation_guess)
            peaks_heights = properties['peak_heights']
            peaks = peaks + index_offset
            df_times[f"region {i}"] = time_fp.iloc[peaks].values
            df_indices[f"region {i}"] = peaks
            df_heights[f"region {i}"] = properties["peak_heights"]

            # region_peaks[f"r{i}"] = {
            #     "time": time_fp[peaks],
            #     "indices": peaks,
            #     "heights": properties["peak_heights"],
            # }

            fig1 = plt.subplot(4,1,i)
            plt.plot(region_time, region_signal, label=f'Region {i} Data', alpha=0.7)
            plt.scatter(region_time[peaks], properties["peak_heights"], color='black', label='Peaks', marker = 'x')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.title(f"Peaks in Region {i}")
            plt.legend()

        #     amp_list, mean_list, width_list, offset_list = [], [], [], []

        #     fig2, ax1 = plt.subplots()
        #     ax2 = ax1.twiny()
            
        #     for idx, peak_idx in enumerate(peaks):
        #         window_size = window_size_input
        #         start = max(peak_idx - window_size, 0)
        #         end = min(peak_idx + window_size, len(region_signal))
        #         region_x = np.arange(start, end)
        #         region_y = region_signal[start:end]

        #         # amp, mean, width, offset
        #         initial_guess = [peaks_heights[peaks.tolist().index(peak_idx)], peak_idx, 5, np.min(region_y)]

        #         popt, cov = curve_fit(lorentzian_with_offset, region_x, region_y, p0=initial_guess)
        #         fitted_curve = lorentzian_with_offset(region_x, *popt)

        #         # amp_list.append(popt[0])
        #         # mean_list.append(popt[1])
        #         # width_list.append(popt[2])
        #         # offset_list.append(popt[3])

        #         if idx == 0:
        #             ax2.plot(region_x, fitted_curve, color='green', linestyle='--', label='Lorentzian Fit')
        #             # plt.plot(region_x, fitted_curve, color='green', linestyle='--')
        #         else:
        #             ax2.plot(region_x, fitted_curve, color='green', linestyle='--')
        #             # plt.plot(region_x, fitted_curve, color='green', linestyle='--')

        #     # plt.plot(time_fp, fp_channel, label='Signal', alpha = 0.4, lw = 0.7)
        #     # plt.scatter(peaks_indices, peaks_heights, color='red', label='Maxima Peaks', zorder=5, marker='x')
        #     # plt.title('Signal with Lorentzian Fits')
        #     # plt.xlabel('Peak Index')
        #     # plt.ylabel('Voltage (V)')
        #     # plt.xticks(peaks_indices, range(1, len(peaks_indices) + 1))
        #     # plt.legend()
        #     # plt.show()

        #     ax1.plot(region_time, region_signal, label='Signal', alpha = 0.4, lw = 0.7)
        #     ax1.set_xlabel('Time (ms)')
        #     ax1.set_ylabel('Voltage (V)')

        #     ax2.scatter(peaks, peaks_heights, color='red', label='Lorentzian Peaks', zorder=5, marker='x', s = 20, linewidths=0.5)
        #     ax2.set_xlabel('Peak Index')
        #     # ax2.set_xticks(peaks, range(1, len(peaks) + 1))    

        #     plt.title('Signal with Lorentzian Fits')
        #     handles, labels = ax1.get_legend_handles_labels()
        #     handles2, labels2 = ax2.get_legend_handles_labels()
        #     handles.extend(handles2)
        #     labels.extend(labels2)
        #     plt.legend(handles=handles, labels=labels, loc='best')
        #     plt.show()
        # print(df_times)
        # print(df_heights)
        # print(df_indices)



        region1_df = pd.DataFrame()
        region2_df = pd.DataFrame()
        region3_df = pd.DataFrame()
        region4_df = pd.DataFrame()

        # List to hold references to the DataFrame variables
        region_dfs = [region1_df, region2_df, region3_df, region4_df]

        # Loop over the regions and assign DataFrames
        for i, (start, end) in enumerate(regions, start=1):
            # Create mask and extract time and signal
            region_mask = (time_fp >= start) & (time_fp <= end)
            region_time = time_fp[region_mask]
            region_signal = fp_channel[region_mask]
            
            # Create a DataFrame for the current region
            region_df = pd.DataFrame({
                'Time': region_time.values,  # Extract values to avoid index complications
                'Signal': region_signal.values
            })
            region_dfs[i - 1] = region_df


        plt.figure(figsize=(10, 8))
        for i, region_df in enumerate(region_dfs, start=1):
            plt.subplot(1, 4, i)
            plt.plot(region_df['Time'], region_df['Signal'], label=f'Region {i}', alpha=0.7)
            plt.title(f'Region {i}')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal')
            plt.legend()
        plt.tight_layout()
        plt.show()








# need to do lorentzian fits here and then get differences between means of peaks
# them find average of all of them and get fsr?
# but do we need to get average or just analyse them separetly as they each are for each hfs splitting state/transition



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

params = dec_5_data_params

analysis(*params, all_plots=False)

# analysis(hfs_file, fs_file, fp_file, peak_height_guess, peak_separation_guess, window_size_input, \
#              all_plots = False, subtracted_manual = False, subtracted_automatic = True, fabry_perot = True, fp_region_wise = True):

# dec_6_data_params = [hfs_file, fs_file, fp_file, peak_height_guess, peak_separation_guess, window_size_input]
