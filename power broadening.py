import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import scipy.optimize as op # type: ignore
from pathlib import Path
from scipy.odr import * # type: ignore

plt.rcParams["figure.figsize"] = (10,6)

# Parameters
angles = np.arange(0, 200, 60)
# angles = [50, 150, 200]
# angles = [150]

def gaussian(x, amp, mu, sigma, c):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c


def four_gaussians(x, amp1, mu1, sigma1, c1, amp2, mu2, sigma2, c2, amp3, mu3, sigma3, c3, amp4, mu4, sigma4, c4):
    return (
        amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) + c1 +
        amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) + c2 +
        amp3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2)) + c3 +
        amp4 * np.exp(-((x - mu4) ** 2) / (2 * sigma4 ** 2)) + c4
    )


def data_get_and_data_go(angles = angles):
    # power_file = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\11 dec data\power data.csv"
    # bg_file = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\11 dec data\BG.csv"
    bg_file = r"D:\maanas lab work\12 dec\BG.csv"
    bg_df = pd.read_csv(bg_file)
    bg_signal = bg_df["C2 in V"]
    bg_time = bg_df["in s"]

    file_path = Path(r"D:\maanas lab work\12 dec")

    df = pd.DataFrame()
    residue_df = pd.DataFrame()



    for angle in angles:
        diff_list = []
        file = file_path/f"{angle}.csv"
        print(file)

        data = pd.read_csv(file)
        signal = data["C2 in V"]
        signal_time = data["in s"]
        df[f"{angle}"] = data["C2 in V"]

        bg_subtracted_signal = signal - bg_signal

        # residue_df[f"{angle}"] = bg_subtracted_signal

        diff_scaled = (bg_subtracted_signal - np.min(bg_subtracted_signal)) / (np.max(bg_subtracted_signal) - np.min(bg_subtracted_signal))
        sig_scaled = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        bg_scaled = (bg_signal - np.min(bg_signal)) / (np.max(bg_signal) - np.min(bg_signal))

        # plt.subplot(3, 1, i+1)
        # plt.plot(diff_scaled, label = 'subtracted')
        # plt.plot(bg_scaled, label = 'bg')
        # # plt.plot(time+.00003, sig_scaled, label = 'signal')
        # plt.plot(sig_scaled, label = 'signal')

        plt.plot(bg_time, bg_subtracted_signal, label = 'subtracted - data')
        plt.plot(bg_time, bg_signal, label = 'bg')
        # plt.plot(time+.00003, sig_scaled, label = 'signal')
        plt.plot(signal_time, signal, label = 'signal')

        # p0 = [-0.400, 0.002, 0.001, 0.00, 
        #   -1.005, 0.008, 0.001, 0.10, 
        #   -0.343, 0.020, 0.001, 0.10, 
        #   -0.40, 0.0305, 0.0015, 0.70          
        # ] # 28 nov data - doppler broadened 1

        # amp, mu, sigma, c
        # p0 = [-0.65, -0.0155, 0.001, 0.2]
        # p0 = [-1.4, -0.0105, 0.0009, 0.2]
        # p0 = [-0.6, 0.0013, 0.001, 0.2]
        # p0 = [-0.24, 0.011, 0.00091, 0.18]

        # plt.plot(time, gaussian(time, *p0), label = 'guess')

        p12 = [0.5, -0.014, 0.001, -0.4]
        fit1, cov1 = op.curve_fit(gaussian, signal_time, signal, p12, maxfev = 1000000)
        # plt.plot(signal_time, gaussian(signal_time, *p12), label = 'guess - signal')
        plt.plot(signal_time, gaussian(signal_time, *fit1), label = 'fit1 - signal')

        fit2, cov2 = op.curve_fit(gaussian, bg_time, bg_signal, p12)
        plt.plot(bg_time, gaussian(bg_time, *fit2), label = 'fit2 - bg')

        # new_sub = signal - gaussian(time, *fit)

        # plt.plot(time, new_sub, label = 'subtracted - fit')
        # plt.xlim(-0.0116, -0.0102)
        # plt.xlim(-0.0120, -0.0095)

        # plt.title(angle)
        # plt.legend()
        # plt.show()

        # print(fit1)
        # print(fit2)

        diff_in_mean_time_12 = np.abs(fit2[1] - fit1[1])
        #shift signal by difference and plot
        diff_list.append(diff_in_mean_time_12)

        
        #%%


        p34 = [-0.65, -0.0155, 0.001, 0.2]
        fit3, cov3 = op.curve_fit(gaussian, signal_time, signal, p34, maxfev = 1000000)
        # plt.plot(signal_time, gaussian(signal_time, *p34), label = 'guess - signal')
        plt.plot(signal_time, gaussian(signal_time, *fit3), label = 'fit3 - signal')

        fit4, cov4 = op.curve_fit(gaussian, bg_time, bg_signal, p34)
        plt.plot(bg_time, gaussian(bg_time, *fit4), label = 'fit4 - bg')

        # new_sub = signal - gaussian(time, *fit)

        # plt.plot(time, new_sub, label = 'subtracted - fit')
        # plt.xlim(-0.0116, -0.0102)
        # plt.xlim(-0.0120, -0.0095)

        # plt.title(angle)
        # plt.legend()
        # plt.show()

        # print(fit1)
        # print(fit2)

        diff_in_mean_time_34 = np.abs(fit4[1] - fit3[1])
        #shift signal by difference and plot
        diff_list.append(diff_in_mean_time_34)

        
        #%%


        p56 = [-1.4, -0.0105, 0.0009, 0.2]
        fit5, cov5 = op.curve_fit(gaussian, signal_time, signal, p56, maxfev = 1000000)
        # plt.plot(signal_time, gaussian(signal_time, *p56), label = 'guess - signal')
        plt.plot(signal_time, gaussian(signal_time, *fit5), label = 'fit5 - signal')

        fit6, cov6 = op.curve_fit(gaussian, bg_time, bg_signal, p56)
        plt.plot(bg_time, gaussian(bg_time, *fit6), label = 'fit6 - bg')

        # new_sub = signal - gaussian(time, *fit)

        # plt.plot(time, new_sub, label = 'subtracted - fit')
        # plt.xlim(-0.0116, -0.0102)
        # plt.xlim(-0.0120, -0.0095)

        # plt.title(angle)
        # plt.legend()
        # plt.show()

        # print(fit1)
        # print(fit2)

        diff_in_mean_time_56 = np.abs(fit6[1] - fit5[1])
        #shift signal by difference and plot
        diff_list.append(diff_in_mean_time_56)


        p78 = [-0.6, 0.0013, 0.001, 0.2]
        fit7, cov7 = op.curve_fit(gaussian, signal_time, signal, p78, maxfev = 1000000)
        # plt.plot(signal_time, gaussian(signal_time, *p78), label = 'guess - signal')
        plt.plot(signal_time, gaussian(signal_time, *fit7), label = 'fit7 - signal')

        fit8, cov8 = op.curve_fit(gaussian, bg_time, bg_signal, p78)
        plt.plot(bg_time, gaussian(bg_time, *fit8), label = 'fit8 - bg')

        # new_sub = signal - gaussian(time, *fit)

        # plt.plot(time, new_sub, label = 'subtracted - fit')
        # plt.xlim(-0.0116, -0.0102)
        # plt.xlim(-0.0120, -0.0095)

        # plt.title(angle)
        # plt.legend()
        # plt.show()

        # print(fit1)
        # print(fit2)

        diff_in_mean_time_78 = np.abs(fit8[1] - fit7[1])
        #shift signal by difference and plot
        diff_list.append(diff_in_mean_time_78)

        mean_diff_in_mean_time = np.average(diff_list)
        print(diff_list)
        print(mean_diff_in_mean_time)


        plt.title(angle)
        plt.legend()
        plt.show()


        signal_time_78 = signal_time + diff_in_mean_time_78
        

        plt.plot(bg_time, bg_signal, label = 'bg')
        # plt.plot(time+.00003, sig_scaled, label = 'signal')
        # plt.plot(signal_time_78, signal, label = 'signal - 78')


        # need to crop old and new bg and signals datasets - start from right offset value (signal) 
        # and end at left most value (bg)

        print(signal_time_78.iloc[0])
        print(bg_time.iloc[-1])
        plt.legend()
        plt.title('shifted - ' + str(angle) + ' degrees')
        plt.show()

        desired_start = signal_time_78.iloc[0]

        closest_index_start = (bg_time - desired_start).abs().idxmin()
        cropped_bg_time = bg_time.loc[closest_index_start:].reset_index(drop=True)
        cropped_bg_signal = bg_signal.loc[closest_index_start:].reset_index(drop=True)


        desired_end = bg_time.iloc[-1]

        closest_index_end = (signal_time_78 - desired_end).abs().idxmin()
        cropped_signal_time = signal_time_78.loc[:closest_index_end].reset_index(drop=True)
        cropped_signal = signal.loc[:closest_index_end].reset_index(drop=True)

        plt.plot(cropped_bg_time, cropped_bg_signal, label = 'bg cropped')
        plt.plot(cropped_signal_time, cropped_signal, label = 'signal cropped - 78')

        # print(len(cropped_bg_time))
        print(len(cropped_bg_signal))
        # print(len(cropped_signal_time))
        print(len(cropped_signal))
        print(bg_signal)
        print(cropped_signal)

        new_subtracted_signal = cropped_signal - cropped_bg_signal
        print(len(new_subtracted_signal))
        time_residue = cropped_signal_time - cropped_bg_time
        print(time_residue.unique())

        plt.plot(bg_time, bg_subtracted_signal, label = 'subtracted signal - old')

        plt.plot(cropped_signal_time, new_subtracted_signal, label = 'subtracted signal - new - 78')











        signal_time_12 = signal_time + diff_in_mean_time_12
        

        print(signal_time_12.iloc[0])
        print(bg_time.iloc[-1])

        desired_start = signal_time_12.iloc[0]

        closest_index_start = (bg_time - desired_start).abs().idxmin()
        cropped_bg_time = bg_time.loc[closest_index_start:].reset_index(drop=True)
        cropped_bg_signal = bg_signal.loc[closest_index_start:].reset_index(drop=True)

        # plt.plot(signal_time_12, signal, label = 'signal - 78')

        desired_end = bg_time.iloc[-1]

        closest_index_end = (signal_time_12 - desired_end).abs().idxmin()
        cropped_signal_time = signal_time_12.loc[:closest_index_end].reset_index(drop=True)
        cropped_signal = signal.loc[:closest_index_end].reset_index(drop=True)

        plt.plot(cropped_bg_time, cropped_bg_signal, label = 'bg cropped')
        plt.plot(cropped_signal_time, cropped_signal, label = 'signal cropped - 12')

        # print(len(cropped_bg_time))
        print(len(cropped_bg_signal))
        # print(len(cropped_signal_time))
        print(len(cropped_signal))
        print(bg_signal)
        print(cropped_signal)

        new_subtracted_signal = cropped_signal - cropped_bg_signal
        print(len(new_subtracted_signal))
        time_residue = cropped_signal_time - cropped_bg_time
        print(time_residue.unique())

        plt.plot(bg_time, bg_subtracted_signal, label = 'subtracted signal - old')

        plt.plot(cropped_signal_time, new_subtracted_signal, label = 'subtracted signal - new - 12')







        signal_time_mean = signal_time + mean_diff_in_mean_time
        



        # need to crop old and new bg and signals datasets - start from right offset value (signal) 
        # and end at left most value (bg)

        print(signal_time_mean.iloc[0])
        print(bg_time.iloc[-1])

        desired_start = signal_time_mean.iloc[0]

        closest_index_start = (bg_time - desired_start).abs().idxmin()
        cropped_bg_time = bg_time.loc[closest_index_start:].reset_index(drop=True)
        cropped_bg_signal = bg_signal.loc[closest_index_start:].reset_index(drop=True)


        desired_end = bg_time.iloc[-1]

        closest_index_end = (signal_time_mean - desired_end).abs().idxmin()
        cropped_signal_time = signal_time_mean.loc[:closest_index_end].reset_index(drop=True)
        cropped_signal = signal.loc[:closest_index_end].reset_index(drop=True)

        plt.plot(cropped_bg_time, cropped_bg_signal, label = 'bg cropped mean')
        plt.plot(cropped_signal_time, cropped_signal, label = 'signal cropped mean')

        # print(len(cropped_bg_time))
        print(len(cropped_bg_signal))
        # print(len(cropped_signal_time))
        print(len(cropped_signal))
        print(bg_signal)
        print(cropped_signal)

        new_subtracted_signal = cropped_signal - cropped_bg_signal
        print(len(new_subtracted_signal))
        time_residue = cropped_signal_time - cropped_bg_time
        print(time_residue.unique())

        plt.plot(cropped_signal_time, new_subtracted_signal, label = 'subtracted signal - new - mean')


        plt.legend()
        plt.title('shifted and cropped - ' + str(angle) + ' degrees')
        plt.show()

        residue_df[f"{angle}"] = new_subtracted_signal



    residue_array = residue_df.to_numpy()
    return residue_df, residue_array


df, arr = data_get_and_data_go()
# np.savetxt("residue.txt", arr)


# print(arr)

plt.plot(arr, label = angles, lw = .2)
plt.legend()
plt.show()
