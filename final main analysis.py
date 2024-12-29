import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op
from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import scipy.constants as const

plt.rcParams["figure.figsize"] = (10, 6)

angles = np.arange(0, 200, 60)
angles = [180]

def gaussian(x, amp, mu, sigma, c):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c

def n_gaussians(x, params):
    if len(params) % 4 != 0:
        raise ValueError("Number of parameters must be a multiple of 4 (amp, mu, sigma, c for each Gaussian).")
    result = np.zeros_like(x, dtype=float)
    for amp, mu, sigma, c in params:
        result += gaussian(x, amp, mu, sigma, c)
    return result

def lorentzian_with_offset(x, amp, mean, width, offset):
    return amp / (1 + ((x - mean) / width) ** 2) + offset

def n_lorentzians(x, *params):
    if len(params) % 4 != 0:
        raise ValueError("Number of parameters must be a multiple of 4 (amp, mean, width, offset for each Lorentzian).")
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        amp, mean, width, offset = params[i:i+4]
        result += lorentzian_with_offset(x, amp, mean, width, offset)
    return result

def min_max_scaling(input_series):
    return (input_series - np.min(input_series)) / (np.max(input_series) - np.min(input_series))

def etalon_theoretical_calc(L = 0.21):
    c = const.c
    fsr = c / (4 * L)
    fsr = fsr * 1e-6
    return fsr

print(etalon_theoretical_calc())

def perform_gaussian_fits(x, y, bg_x, bg_y, initial_params_list):
    diffs = []
    fits = []

    for params in initial_params_list:
        fit_signal, _ = op.curve_fit(gaussian, x, y, params, maxfev=1000000)
        fit_bg, _ = op.curve_fit(gaussian, bg_x, bg_y, params)
        fits.append((fit_signal, fit_bg))

        diff_in_mean_time = np.abs(fit_bg[1] - fit_signal[1])
        diffs.append(diff_in_mean_time)

    mean_diff_time = np.average(diffs)
    return mean_diff_time, fits

def plot_gaussians(x, y, bg_x, bg_y, fits, angle):
    plt.plot(bg_x, bg_y, label='Background')
    plt.plot(x, y, label='Signal')

    for i, (fit_signal, fit_bg) in enumerate(fits):
        plt.plot(x, gaussian(x, *fit_signal), label=f'Fit {i + 1} - Signal')
        plt.plot(bg_x, gaussian(bg_x, *fit_bg), label=f'Fit {i + 1} - Background')

    plt.title(f'Gaussian Fits for {angle} Degrees')
    plt.legend()
    plt.show()

def data_cleaning(angles=angles, plot=True):
    bg_file = r"D:\maanas lab work\12 dec\BG.csv"
    bg_df = pd.read_csv(bg_file)
    bg_signal = bg_df["C2 in V"]
    bg_time = bg_df["in s"]

    file_path = Path(r"D:\maanas lab work\12 dec")
    residue_df = pd.DataFrame()
    fp_df = pd.DataFrame()

    initial_params_list = [
        [0.5, -0.014, 0.001, -0.4],
        [-0.65, -0.0155, 0.001, 0.2],
        [-1.4, -0.0105, 0.0009, 0.2],
        [-0.6, 0.0013, 0.001, 0.2]
    ]

    i = 0
    for angle in angles:
        file = file_path / f"{angle}.csv"
        print(file)

        data = pd.read_csv(file)
        signal = data["C2 in V"]
        signal_time = data["in s"]

        mean_diff_time, fits = perform_gaussian_fits(signal_time, signal, bg_time, bg_signal, initial_params_list)

        if plot:
            plot_gaussians(signal_time, signal, bg_time, bg_signal, fits, angle)


        signal_time_shifted = signal_time + mean_diff_time
        desired_start = signal_time_shifted.iloc[0]

        closest_index_start = (bg_time - desired_start).abs().idxmin()
        cropped_bg_time = bg_time.loc[closest_index_start:].reset_index(drop=True)
        cropped_bg_signal = bg_signal.loc[closest_index_start:].reset_index(drop=True)

        desired_end = bg_time.iloc[-1]
        closest_index_end = (signal_time_shifted - desired_end).abs().idxmin()
        cropped_signal_time = signal_time_shifted.loc[:closest_index_end].reset_index(drop=True)
        cropped_signal = signal.loc[:closest_index_end].reset_index(drop=True)

        new_subtracted_signal = cropped_signal - cropped_bg_signal
        time_residue = cropped_signal_time - cropped_bg_time

        fp_data = data["C3 in V"]
        cropped_fp = fp_data.loc[:closest_index_end].reset_index(drop=True)


        if plot:
            plt.plot(cropped_signal_time, cropped_bg_signal, label='Cropped Background')
            plt.plot(cropped_signal_time, cropped_signal, label='Cropped Signal')
            plt.plot(cropped_signal_time, new_subtracted_signal, label='Subtracted Signal - New')
            plt.plot(cropped_signal_time, cropped_fp, label = 'Cropped Fabry Perot')
            plt.legend()
            plt.title(f'Shifted and Cropped - {angle} Degrees')
            plt.show()

        base_time = cropped_signal_time - cropped_signal_time[0]
        print(base_time)
        if i == 0:
            residue_df[f"Reference Time"] = base_time

        residue_df[f"{angle}"] = new_subtracted_signal
        fp_df[f"{angle}"] = cropped_fp

        i = 1

    residue_array = residue_df.to_numpy()
    return residue_df, fp_df, residue_array

subtracted_df, fp_df, arr = data_cleaning(plot=False)
# np.savetxt("residue.txt", arr)

# print(subtracted_df)
# print(subtracted_df.isna().sum())
# print(fp_df)
# print(arr)

def fp_analysis(angles = angles, subtracted_df = subtracted_df, fp_df = fp_df):
    for angle in angles:
        time = subtracted_df["Reference Time"]
        fp_data = fp_df[f"{angle}"]
        fp_data_scaled = (fp_data - np.min(fp_data)) / (np.max(fp_data) - np.min(fp_data))

        subtracted_signal = subtracted_df[f"{angle}"]
        signal_scaled = (subtracted_signal - np.min(subtracted_signal)) / (np.max(subtracted_signal) - np.min(subtracted_signal))
        plt.plot(time, fp_data_scaled, label = 'Scaled Fabry Perot Spectrum')
        plt.plot(time, signal_scaled, label = 'Scaled Subtracted Spectrum')
        plt.legend()
        plt.title(f"Fabry Perot Analysis for {angle} Degrees")
        plt.show()
        
        r1_start = 0.000
        r1_end = 0.0049
        r2_start = 0.0051
        r2_end = 0.010
        r3_start = 0.0175
        r3_end = 0.022
        r4_start = 0.028
        r4_end = 0.03228

        r_start = 0.003
        r_end = 0.024

        # all plots overlayed
        # plt.plot(time, fs_channel/np.max(fs_channel), label = 'FS Data')
        # plt.plot(time, hfs_channel/np.max(hfs_channel), label = 'HFS Data')
        plt.plot(time, signal_scaled, label = 'Scaled Subtracted Spectrum', alpha = 0.6)
        plt.plot(time, fp_data_scaled, label = 'Scaled Fabry Perot Spectrum', color = 'black', alpha = 0.3, lw = 0.5)
        plt.fill_betweenx([0, 1], r1_start, r1_end, color='gray', alpha=0.25, label="R1")
        plt.fill_betweenx([0, 1], r2_start, r2_end, color='gray', alpha=0.40, label="R2")
        plt.fill_betweenx([0, 1], r3_start, r3_end, color='gray', alpha=0.55, label="R3")
        plt.fill_betweenx([0, 1], r4_start, r4_end, color='gray', alpha=0.70, label="R4")

        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Selected Regions for {angle} degrees')
        plt.legend()
        plt.show()

        regions = [(r1_start, r1_end), (r2_start, r2_end), \
                (r3_start, r3_end), (r4_start, r4_end)]

        # regions = [(r_start, r_end)]

        region_peaks = {}
        index_offset = 0


        df_times = pd.DataFrame()
        df_indices = pd.DataFrame()
        df_heights = pd.DataFrame()


        peak_height_guess = 0.001
        peak_separation_guess = 5000
        window_size_input = 2000

        region1_df = pd.DataFrame()
        region2_df = pd.DataFrame()
        region3_df = pd.DataFrame()
        region4_df = pd.DataFrame()

        region1_param_df = pd.DataFrame()
        region2_param_df = pd.DataFrame()
        region3_param_df = pd.DataFrame()
        region4_param_df = pd.DataFrame()

        region1_peak_params_df = pd.DataFrame()
        region2_peak_params_df = pd.DataFrame()
        region3_peak_params_df = pd.DataFrame()
        region4_peak_params_df = pd.DataFrame()

        region_dfs = [region1_df, region2_df, region3_df, region4_df]
        region_params_dfs = [region1_param_df, region2_param_df, region3_param_df, region4_param_df]
        region_peak_params_dfs = [region1_peak_params_df, region2_peak_params_df, region3_peak_params_df, region4_peak_params_df]

        columns = ["Amp", "Amp Error", "Mean", "Mean Error", 
           "Std Dev", "Std Dev Error", "Offset", "Offset Error"]
        
        for df in region_params_dfs:
            df = pd.DataFrame(columns=columns)

        for i, (start, end) in enumerate(regions, start = 1):
            region_mask = (time >= start) & (time <= end)    
            index_offset = (time - start).abs().idxmin()
            region_time = time[region_mask]
            region_fp_signal = fp_data[region_mask]
            region_signal = signal_scaled[region_mask]

            region_df = pd.DataFrame({
                'Time': region_time.values,
                'Signal': region_signal.values,
                'FP Signal': region_fp_signal.values
            })
            region_dfs[i - 1] = region_df

            peaks, properties = find_peaks(region_fp_signal, height=peak_height_guess, distance=peak_separation_guess)
            peaks_heights = properties['peak_heights']
            peaks = peaks + index_offset
            # print(i)
            # print(peaks)

            df_times[f"region {i}"] = time.iloc[peaks].values
            df_indices[f"region {i}"] = peaks
            df_heights[f"region {i}"] = properties["peak_heights"]

            for peak_idx, peak_height in zip(peaks, peaks_heights):
                peak_params = {
                    "Index": peak_idx,
                    "Time": time.iloc[peak_idx],
                    "Height": peak_height
                }
                region_peak_params_dfs[i - 1] = pd.concat(
                    [region_peak_params_dfs[i - 1], pd.DataFrame([peak_params])], 
                    ignore_index=True)

            # region_peaks[f"r{i}"] = {
            #     "time": time_fp[peaks],
            #     "indices": peaks,
            #     "heights": properties["peak_heights"],
            # }

            # fig1 = plt.subplot(4,1,i)
            # plt.plot(region_time, region_fp_signal, label=f'Region {i} Scaled Fabry Perot Data', color = 'black', alpha = 0.4)
            # print(region_time[peaks])
            # # plt.plot(region_time, region_signal, label=f'Region {i} Scaled Subtracted Spectrum Data', alpha = 0.6)
            # plt.scatter(region_time[peaks], properties["peak_heights"], color='black', label='Peaks', marker = 'x')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Signal')
            # plt.title(f"Peaks in Region {i}")
            # plt.legend()
            # plt.tight_layout()
        

            amp_list, mean_list, width_list, offset_list = [], [], [], []

            fig2 = plt.subplot(4,1,i)
            # fig2, ax1 = plt.subplots()
            # ax2 = ax1.twiny()
            
            for idx, peak_idx in enumerate(peaks):
                print(idx)
                window_size = window_size_input
                start = max(peak_idx - window_size, 0)
                end = min(peak_idx + window_size, len(fp_data))
                print(start, end)
                region_x = np.arange(start, end)
                region_x_time = region_time.loc[start:end-1]
                print(region_fp_signal)
                region_y = region_fp_signal.loc[start:end-1]
                # print(region_x)
                # print(region_y)

                # amp, mean, width, offset
                initial_guess = [peaks_heights[peaks.tolist().index(peak_idx)], peak_idx, 5, np.min(region_y)]

                popt, cov = op.curve_fit(lorentzian_with_offset, region_x, region_y, p0=initial_guess)
                fitted_curve = lorentzian_with_offset(region_x, *popt)

                print("params:", popt)
                print("cov mat:", cov)
                
                params = {
                "Amplitude": popt[0],
                "Amp Error": np.sqrt(cov[0,0]),
                "Mean": popt[1],
                "Mean Error": np.sqrt(cov[1,1]),
                "Std Dev": popt[2],
                "Std Dev Error": np.sqrt(cov[2,2]),
                "Offset": popt[3],
                "Offset Error": np.sqrt(cov[3,3])
                }

                region_params_dfs[i-1] = pd.concat([region_params_dfs[i-1], pd.DataFrame([params])], ignore_index=True)

                if idx == 0:
                    # ax2.plot(region_x, fitted_curve, color='green', linestyle='--', label='Lorentzian Fit')
                    plt.plot(region_x_time, fitted_curve, color='green', linestyle='--', label='Lorentzian Fit')
                else:
                    # ax2.plot(region_x, fitted_curve, color='green', linestyle='--')
                    plt.plot(region_x_time, fitted_curve, color='green', linestyle='--')

            print("region_params_dfs")
            print(region_params_dfs[i-1])
            plt.plot(region_time, region_fp_signal, label='Fabry Perot Signal', alpha = 0.4, lw = 0.7)
            # plt.scatter(peaks_indices, peaks_heights, color='red', label='Maxima Peaks', zorder=5, marker='x')
            plt.scatter(region_time[peaks], peaks_heights, color='black', label='Peaks', marker = 'x')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.title(f"{angle} Degrees - Region {i}")
            # plt.xticks(region_time[peaks], range(1, len(region_time[peaks]) + 1))
            plt.legend()

            # ax1.plot(region_time, region_fp_signal, label='FP Signal', alpha = 0.4, lw = 0.7)
            # ax1.set_xlabel('Time (ms)')
            # ax1.set_ylabel('Voltage (V)')

            # ax2.scatter(peaks, peaks_heights, color='red', label='Lorentzian Peaks', zorder=5, marker='x', s = 20, linewidths=0.5)
            # ax2.set_xlabel('Peak Index')
            # # ax2.set_xticks(peaks, range(1, len(peaks) + 1))    

            # plt.title('Signal with Lorentzian Fits')
            # handles, labels = ax1.get_legend_handles_labels()
            # handles2, labels2 = ax2.get_legend_handles_labels()
            # handles.extend(handles2)
            # labels.extend(labels2)
            # plt.legend(handles=handles, labels=labels, loc='best')
            plt.suptitle(f'Signal with Lorentzian Fits for {angle} Degrees')
        plt.show()
    print("---" * 30)
    print(df_heights)
    print(df_indices)
    print(df_times)
    print(df_times.diff(axis=0))
    print(df_times.diff(axis=0).mean())
    print(etalon_theoretical_calc())
    conversion_factor = etalon_theoretical_calc() / df_times.diff(axis=0).mean()
    print("MHz / sec:", conversion_factor)
    # print(region_dfs)
    # print("region_params_dfs")
    # print(region_params_dfs)
    print(region_peak_params_dfs)

    time_conversion_factor_df = pd.DataFrame()
    index_conversion_factor_df = pd.DataFrame()
    theory = etalon_theoretical_calc()
    for i, df in enumerate(region_peak_params_dfs):
        print(f"Angle {angle}, Region {i}")
        times = df["Time"]
        time_diff = times.diff()
        mean_time_diff = time_diff.mean()
        # print(mean_time_diff)
        time_conversion_factor = theory / mean_time_diff
        print(time_conversion_factor)
        time_conversion_factor_df.loc[i, f"{angle}"] = time_conversion_factor

        indices = df["Index"]
        index_diff = indices.diff()
        mean_index_diff = index_diff.mean()
        # print(mean_time_diff)
        index_conversion_factor = theory / mean_index_diff
        print(index_conversion_factor)
        index_conversion_factor_df.loc[i, f"{angle}"] = index_conversion_factor        

    return region_dfs, region_params_dfs, region_peak_params_dfs, time_conversion_factor_df, index_conversion_factor_df


region_dfs, region_params_dfs, region_peak_params_dfs, time_conversion_factor_df, index_conversion_factor_df = fp_analysis()

print(time_conversion_factor_df)
print(index_conversion_factor_df)

def fp_conversion(region_peak_params_dfs):
    theory = etalon_theoretical_calc()
    time_conversion_factor_list = []
    index_conversion_factor_list = []
    for i, df in enumerate(region_peak_params_dfs):
        print(f"Region {i}")
        times = df["Time"]
        time_diff = times.diff()
        mean_time_diff = time_diff.mean()
        # print(mean_time_diff)
        time_conversion_factor = theory / mean_time_diff
        print(time_conversion_factor)
        time_conversion_factor_list.append(time_conversion_factor)

        indices = df["Index"]
        index_diff = indices.diff()
        mean_index_diff = index_diff.mean()
        # print(mean_time_diff)
        index_conversion_factor = theory / mean_index_diff
        print(index_conversion_factor)
        index_conversion_factor_list.append(index_conversion_factor)


    print("MHz / sec:", time_conversion_factor_list)
    print("MHz / index:", index_conversion_factor_list)
    return time_conversion_factor_list, index_conversion_factor_list

# time_conversion_factor_list, index_conversion_factor_list = fp_conversion()

def hfs_plots():
    print("hfs_plots")
    for angle in angles:
        if angle == 180:
            r1_plot_df, r2_plot_df, r3_plot_df, r4_plot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            plotting_dfs = [r1_plot_df, r2_plot_df, r3_plot_df, r4_plot_df]
            time = subtracted_df["Reference Time"]
            indices = subtracted_df.index.values
            signal = subtracted_df[f"{angle}"]
            time_conversion_factors = time_conversion_factor_df[f"{angle}"]
            index_conversion_factors = index_conversion_factor_df[f"{angle}"]

            # freq_from_time = time * time_conversion_factors.mean()
            # freq_from_index = indices * index_conversion_factors.mean()
            
            # plt.plot(freq_from_time, signal, label = 'time')
            # plt.plot(freq_from_index, signal, label = 'index')
            # plt.legend()
            # plt.show()

            for i, df in enumerate(region_dfs, start = 1):
                plt.subplot(4, 1, i)
                regional_time = df["Time"]
                regional_indices = df.index.values
                regional_signal = df["Signal"]
                regional_fp_signal = df["FP Signal"]
                regional_conv_fact_time = time_conversion_factors[i-1]
                regional_conv_fact_index = index_conversion_factors[i-1]
                freq_from_time = regional_time * regional_conv_fact_time
                freq_from_index = regional_indices * regional_conv_fact_index
                # plt.plot(regional_time, regional_signal, label = 'org tim')
                # plt.plot(regional_indices, regional_signal, label = 'org ind')

                plt.plot(freq_from_time, regional_signal, label = 'time freq')
                plt.plot(freq_from_index, regional_signal, label = 'ind freq')
                dummy_df = pd.DataFrame({
                'Freq_Time': freq_from_time,
                'Freq_Index': freq_from_index,
                'Signal': regional_signal
                })
                
                plotting_dfs[i - 1] = dummy_df

                # plt.title(f"Angle {angle} Region {i}")
                plt.title(f"Region {i}")

            plt.suptitle(f"HFS from angle {angle}")
            plt.legend()
            plt.show()

    return plotting_dfs

plotting_dfs = hfs_plots()

r1_plot_df, r2_plot_df, r3_plot_df, r4_plot_df = plotting_dfs
# print(r1_plot_df, r2_plot_df, r3_plot_df, r4_plot_df)
# r1_plot_df.to_csv('r1_plot_df.csv', index=False)
# r2_plot_df.to_csv('r2_plot_df.csv', index=False)
# r3_plot_df.to_csv('r3_plot_df.csv', index=False)
# r4_plot_df.to_csv('r4_plot_df.csv', index=False)

# r1_plot_df = pd.read_csv("r1_plot_df.csv")

# r1_plot_df = pd.read_csv("r1_plot_df.csv")
# r2_plot_df = pd.read_csv("r2_plot_df.csv")
# r3_plot_df = pd.read_csv("r3_plot_df.csv")
# r4_plot_df = pd.read_csv("r4_plot_df.csv")



def nice_hfs_plots(indiv_curve_fit = False, multi_curve_fit = False, \
                   rb87_lower = False, rb85_lower = False, rb85_upper = False, rb87_upper = False):
    # r1_plot_df, r2_plot_df, r3_plot_df, r4_plot_df = plotting_dfs
    # print(r1_plot_df, r2_plot_df, r3_plot_df, r4_plot_df)
    freq_range_plotting = 600

    if rb87_lower is True:
        # rb87 lower
        target_value_start_r1 = 120
        target_value_end_r1 = target_value_start_r1 + freq_range_plotting
        r1_plot_df_sliced = r1_plot_df[
            (r1_plot_df["Freq_Index"] > target_value_start_r1) & 
            (r1_plot_df["Freq_Index"] < target_value_end_r1)
        ]

        value = len(r1_plot_df_sliced) / 100
        window_size = int(value) if int(value) % 2 != 0 else int(value) + 1
        if abs(value - int(value)) >= 0.5:
            window_size += 2 * (window_size > value) - 1

        r1_plot_df_sliced_signal_filtered = savgol_filter(r1_plot_df_sliced["Signal"], window_length = window_size, polyorder = 3, mode = "nearest")

        freq_index_usage = r1_plot_df_sliced["Freq_Index"] - r1_plot_df_sliced["Freq_Index"].iloc[0]

        # plt.subplot(4,1,1)

        plt.plot(freq_index_usage, r1_plot_df_sliced["Signal"], lw = 0.5, label = 'Data')
        plt.plot(freq_index_usage, r1_plot_df_sliced_signal_filtered, label = f'SavGol with Window Size {window_size}')

        if multi_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                0.02, 80, 10, 0.03,
                0.05, 156, 6, 0.03,
                0.03, 232, 6, 0.03,
                0.20, 286, 4, 0.03,
                0.45, 361, 4, 0.03,
                0.03, 492, 6, 0.032 # to fix
                ]

            popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r1_plot_df_sliced_signal_filtered, p0=initial_guesses, maxfev=100000000)
            # popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r1_plot_df_sliced["Signal"], p0=initial_guesses, maxfev=100000000)
            fitted_curve = n_lorentzians(freq_index_usage, *popt)

            param_errors = np.sqrt(np.diag(cov))
            
            params_list = []
            for i in range(0, len(popt), 4):
                params = {
                    "Amplitude": popt[i],
                    "Amp Error": param_errors[i],
                    "Mean": popt[i+1],
                    "Mean Error": param_errors[i+1],
                    "Std Dev": popt[i+2],
                    "Std Dev Error": param_errors[i+2],
                    "Offset": popt[i+3],
                    "Offset Error": param_errors[i+3]
                }
                params_list.append(params)

            rb87_lower_params_df = pd.DataFrame(params_list)

            plt.plot(freq_index_usage, fitted_curve, label = 'Fitted Lorentzians')

        print(rb87_lower_params_df)

        plt.title("Rb 87 Lower")
        plt.legend()
        plt.show()



    if rb85_lower is True:
        # rb85 lower
        target_value_start_r2 = 325
        target_value_end_r2 = target_value_start_r2 + freq_range_plotting
        r2_plot_df_sliced = r2_plot_df[
            (r2_plot_df["Freq_Index"] > target_value_start_r2) & 
            (r2_plot_df["Freq_Index"] < target_value_end_r2)
        ]

        value = len(r2_plot_df_sliced) / 100
        window_size = int(value) if int(value) % 2 != 0 else int(value) + 1
        if abs(value - int(value)) >= 0.5:
            window_size += 2 * (window_size > value) - 1

        r2_plot_df_sliced_signal_filtered = savgol_filter(r2_plot_df_sliced["Signal"], window_length = window_size, polyorder = 3, mode = "nearest")

        freq_index_usage = r2_plot_df_sliced["Freq_Index"] - r2_plot_df_sliced["Freq_Index"].iloc[0]

        # plt.subplot(4,1,1)

        plt.plot(freq_index_usage, r2_plot_df_sliced["Signal"], lw = 0.5, label = 'Data')
        plt.plot(freq_index_usage, r2_plot_df_sliced_signal_filtered, label = f'SavGol with Window Size {window_size}')

        if multi_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                0.02, 200, 20, 10,
                0.10, 234, 8, 0.03,
                0.10, 264, 5, 0.08,
                0.55, 290, 10, -0.001,
                0.80, 320, 13, -0.001,
                0.05, 400, 10, 10 # to fix
                ]
            
            print(len(initial_guesses))

            popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r2_plot_df_sliced_signal_filtered, p0=initial_guesses, maxfev=100000000)
            # popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r2_plot_df_sliced["Signal"], p0=initial_guesses, maxfev=100000000)
            fitted_curve = n_lorentzians(freq_index_usage, *popt)

            param_errors = np.sqrt(np.diag(cov))
            
            params_list = []
            for i in range(0, len(popt), 4):
                params = {
                    "Amplitude": popt[i],
                    "Amp Error": param_errors[i],
                    "Mean": popt[i+1],
                    "Mean Error": param_errors[i+1],
                    "Std Dev": popt[i+2],
                    "Std Dev Error": param_errors[i+2],
                    "Offset": popt[i+3],
                    "Offset Error": param_errors[i+3]
                }
                params_list.append(params)

            rb85_lower_params_df = pd.DataFrame(params_list)

            plt.plot(freq_index_usage, fitted_curve, label = 'Fitted Lorentzians')

        print(rb85_lower_params_df)

        plt.title("Rb 85 Lower")
        plt.legend()
        plt.show()



    if rb85_upper is True:
        # rb85 upper
        target_value_start_r3 = 320
        target_value_end_r3 = target_value_start_r3 + freq_range_plotting
        r3_plot_df_sliced = r3_plot_df[
            (r3_plot_df["Freq_Index"] > target_value_start_r3) & 
            (r3_plot_df["Freq_Index"] < target_value_end_r3)
        ]

        value = len(r3_plot_df_sliced) / 100
        window_size = int(value) if int(value) % 2 != 0 else int(value) + 1
        if abs(value - int(value)) >= 0.5:
            window_size += 2 * (window_size > value) - 1

        r3_plot_df_sliced_signal_filtered = savgol_filter(r3_plot_df_sliced["Signal"], window_length = window_size, polyorder = 3, mode = "nearest")

        freq_index_usage = r3_plot_df_sliced["Freq_Index"] - r3_plot_df_sliced["Freq_Index"].iloc[0]

        # plt.subplot(4,1,1)

        plt.plot(freq_index_usage, r3_plot_df_sliced["Signal"], lw = 0.5, label = 'Data')
        plt.plot(freq_index_usage, r3_plot_df_sliced_signal_filtered, label = f'SavGol with Window Size {window_size}')

        if multi_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                0.065, 240, 1, 30,
                0.10, 259, 2, 30,
                0.15, 273, 10, 30,
                0.25, 292, 4, 30,
                0.20, 299, 8, 30,
                0.08, 329, 15, 30 
                ] # better fitting might be possible? - last lorentian!
            

            # initial_guesses = [
            #     0.065, 240, 1, 0.02,
            #     0.10, 259, 2, 0.08,
            #     0.15, 273, 10, 0.08,
            #     0.25, 292, 4, 0.03,
            #     0.25, 300, 4, 0.04,
            #     0.05, 330, 6, 0.03
            #     ]


            print(len(initial_guesses))

            popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r3_plot_df_sliced_signal_filtered, p0=initial_guesses, maxfev=100000000)
            # popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r3_plot_df_sliced["Signal"], p0=initial_guesses, maxfev=100000000)
            # fitted_curve = n_lorentzians(freq_index_usage, *initial_guesses)
            fitted_curve = n_lorentzians(freq_index_usage, *popt)

            param_errors = np.sqrt(np.diag(cov))
            
            params_list = []
            for i in range(0, len(popt), 4):
                params = {
                    "Amplitude": popt[i],
                    "Amp Error": param_errors[i],
                    "Mean": popt[i+1],
                    "Mean Error": param_errors[i+1],
                    "Std Dev": popt[i+2],
                    "Std Dev Error": param_errors[i+2],
                    "Offset": popt[i+3],
                    "Offset Error": param_errors[i+3]
                }
                params_list.append(params)

            rb85_upper_params_df = pd.DataFrame(params_list)

            plt.plot(freq_index_usage, fitted_curve, label = 'Fitted Lorentzians')

        print(rb85_upper_params_df)

        plt.title("Rb 85 Upper")
        plt.legend()
        plt.show()


    if rb87_upper is True:
        # rb87 upper
        target_value_start_r4 = 150
        target_value_end_r4 = target_value_start_r4 + freq_range_plotting
        r4_plot_df_sliced = r4_plot_df[
            (r4_plot_df["Freq_Index"] > target_value_start_r4) & 
            (r4_plot_df["Freq_Index"] < target_value_end_r4)
        ]

        value = len(r4_plot_df_sliced) / 100
        window_size = int(value) if int(value) % 2 != 0 else int(value) + 1
        if abs(value - int(value)) >= 0.5:
            window_size += 2 * (window_size > value) - 1

        r4_plot_df_sliced_signal_filtered = savgol_filter(r4_plot_df_sliced["Signal"], window_length = window_size, polyorder = 3, mode = "nearest")

        freq_index_usage = r4_plot_df_sliced["Freq_Index"] - r4_plot_df_sliced["Freq_Index"].iloc[0]

        # plt.subplot(4,1,1)

        plt.plot(freq_index_usage, r4_plot_df_sliced["Signal"], lw = 0.5, label = 'Data')
        plt.plot(freq_index_usage, r4_plot_df_sliced_signal_filtered, label = f'SavGol with Window Size {window_size}')

        if multi_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                -0.2, 203, 6, -40,
                0.02, 232, 4, 20,
                0.02, 278, 2, 10,
                0.10, 307, 10, 30,
                0.10, 385, 1, 60
                ]

            print(len(initial_guesses))

            popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r4_plot_df_sliced_signal_filtered, p0=initial_guesses, maxfev=100000000)
            # popt, cov = op.curve_fit(n_lorentzians, freq_index_usage, r4_plot_df_sliced["Signal"], p0=initial_guesses, maxfev=100000000)
            # fitted_curve = n_lorentzians(freq_index_usage, *initial_guesses)
            fitted_curve = n_lorentzians(freq_index_usage, *popt)

            param_errors = np.sqrt(np.diag(cov))
            
            params_list = []
            for i in range(0, len(popt), 4):
                params = {
                    "Amplitude": popt[i],
                    "Amp Error": param_errors[i],
                    "Mean": popt[i+1],
                    "Mean Error": param_errors[i+1],
                    "Std Dev": popt[i+2],
                    "Std Dev Error": param_errors[i+2],
                    "Offset": popt[i+3],
                    "Offset Error": param_errors[i+3]
                }
                params_list.append(params)

            rb87_upper_params_df = pd.DataFrame(params_list)

            plt.plot(freq_index_usage, fitted_curve, label = 'Fitted Lorentzians')

        print(rb87_upper_params_df)

        plt.title("Rb 87 Upper")
        plt.legend()
        plt.show()

nice_hfs_plots(indiv_curve_fit = False, multi_curve_fit = True, \
               rb87_lower = True, rb85_lower = True, rb85_upper = True, rb87_upper = True)

