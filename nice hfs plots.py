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

r1_plot_df = pd.read_csv("r1_plot_df.csv")
r2_plot_df = pd.read_csv("r2_plot_df.csv")
r3_plot_df = pd.read_csv("r3_plot_df.csv")
r4_plot_df = pd.read_csv("r4_plot_df.csv")

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

def nice_hfs_plots(indiv_curve_fit = False, multi_curve_fit = False, rb87_lower = False, rb85_lower = False, rb85_upper = False, rb87_upper = False):
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

        if indiv_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                [0.02, 80, 10, 0.17, 5],   # Window = 5
                [0.05, 156, 6, 0.17, 4],   # Default window = 3
                [0.03, 232, 6, 0.19, 4],   # Window = 1.4
                [0.20, 286, 4, 0.18, 6],   # Default window = 3
                [0.41, 361, 4, 0.21, 6],   # Default window = 3
                [0.02, 492, 6, 0.18, 4]    # Default window = 3
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

                window = (freq_index_usage >= time_min) & (freq_index_usage <= time_max)
                time_window = freq_index_usage[window]
                spectrum_window = r1_plot_df_sliced_signal_filtered[window]

                popt, cov = op.curve_fit(lorentzian_with_offset, time_window, spectrum_window, p0=p0, maxfev=1000000)

                amp, mu, sigma, c = popt
                amplitudes_list.append(amp)
                means_list.append(mu)
                sigmas_list.append(sigma)
                offsets_list.append(c)
                window_times_list.append(time_window)

                fitted_curves_list.append(lorentzian_with_offset(time_window, *popt))

            for i, (fitted_time, fitted_curve) in enumerate(zip(window_times_list, fitted_curves_list), start=1):
                plt.plot(fitted_time, fitted_curve, label = f'L {i}')



        if multi_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                0.02, 80, 10, 0.03,
                0.05, 156, 6, 0.03,
                0.03, 232, 6, 0.03,
                0.20, 286, 4, 0.03,
                0.45, 361, 4, 0.03,
                0.03, 492, 1, 20 # to fix
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

        if indiv_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                [0.05, 234, 10, 0.22, 4],   # Window = 5
                [0.01, 264, 5, 0.33, 4],   # Window = 4
                [0.35, 290, 10, 0.47, 4],   # Window = 4
                [0.50, 320, 13, 0.50, 4],   # Window = 6
                [0.01, 378, 5, 0.19, 4],   # window = 6
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

                window = (freq_index_usage >= time_min) & (freq_index_usage <= time_max)
                time_window = freq_index_usage[window]
                spectrum_window = r2_plot_df_sliced_signal_filtered[window]

                popt, cov = op.curve_fit(lorentzian_with_offset, time_window, spectrum_window, p0=p0, maxfev=1000000)

                amp, mu, sigma, c = popt
                amplitudes_list.append(amp)
                means_list.append(mu)
                sigmas_list.append(sigma)
                offsets_list.append(c)
                window_times_list.append(time_window)

                fitted_curves_list.append(lorentzian_with_offset(time_window, *initial_guesses))

            for i, (fitted_time, fitted_curve) in enumerate(zip(window_times_list, fitted_curves_list), start=1):
                plt.plot(fitted_time, fitted_curve, label = f'L {i}')



        if multi_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                0.02, 202, 8, 0.0,
                0.10, 234, 8, 0.03,
                0.10, 264, 5, 0.08,
                0.55, 290, 10, -0.001,
                0.80, 320, 13, -0.001,
                0.05, 380, 3, 20 # to fix
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

        plt.title("Rb 85 Lower")
        plt.legend()
        plt.show()
        print(rb85_lower_params_df)



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

        if indiv_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                [0.05, 234, 10, 0.22, 4],   # Window = 5
                [0.01, 264, 5, 0.33, 4],   # Window = 4
                [0.35, 290, 10, 0.47, 4],   # Window = 4
                [0.50, 320, 13, 0.50, 4],   # Window = 6
                [0.01, 378, 5, 0.19, 4],   # window = 6
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

                window = (freq_index_usage >= time_min) & (freq_index_usage <= time_max)
                time_window = freq_index_usage[window]
                spectrum_window = r3_plot_df_sliced_signal_filtered[window]

                popt, cov = op.curve_fit(lorentzian_with_offset, time_window, spectrum_window, p0=p0, maxfev=1000000)

                amp, mu, sigma, c = popt
                amplitudes_list.append(amp)
                means_list.append(mu)
                sigmas_list.append(sigma)
                offsets_list.append(c)
                window_times_list.append(time_window)

                fitted_curves_list.append(lorentzian_with_offset(time_window, *initial_guesses))

            for i, (fitted_time, fitted_curve) in enumerate(zip(window_times_list, fitted_curves_list), start=1):
                plt.plot(fitted_time, fitted_curve, label = f'L {i}')



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

        if indiv_curve_fit is True:
            # amp, mean, width, offset
            initial_guesses = [
                [0.05, 203, 10, 0.22, 4],   # Window = 5
                [0.01, 232, 5, 0.33, 4],   # Window = 4
                [0.35, 278, 10, 0.47, 4],   # Window = 4
                [0.50, 307, 13, 0.50, 4],   # Window = 6
                [0.01, 382, 5, 0.19, 4],   # window = 6
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

                window = (freq_index_usage >= time_min) & (freq_index_usage <= time_max)
                time_window = freq_index_usage[window]
                spectrum_window = r4_plot_df_sliced_signal_filtered[window]

                popt, cov = op.curve_fit(lorentzian_with_offset, time_window, spectrum_window, p0=p0, maxfev=1000000)

                amp, mu, sigma, c = popt
                amplitudes_list.append(amp)
                means_list.append(mu)
                sigmas_list.append(sigma)
                offsets_list.append(c)
                window_times_list.append(time_window)

                fitted_curves_list.append(lorentzian_with_offset(time_window, *initial_guesses))

            for i, (fitted_time, fitted_curve) in enumerate(zip(window_times_list, fitted_curves_list), start=1):
                plt.plot(fitted_time, fitted_curve, label = f'L {i}')



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

