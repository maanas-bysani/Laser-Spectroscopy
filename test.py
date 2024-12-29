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
        file = file_path/f"{angle}.csv"
        print(file)

        data = pd.read_csv(file)
        signal = data["C2 in V"]
        signal_time = data["in s"]
        df[f"{angle}"] = data["C2 in V"]

        bg_subtracted_signal = signal - bg_signal

        residue_df[f"{angle}"] = bg_subtracted_signal

        diff_scaled = (bg_subtracted_signal - np.min(bg_subtracted_signal)) / (np.max(bg_subtracted_signal) - np.min(bg_subtracted_signal))
        sig_scaled = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        bg_scaled = (bg_signal - np.min(bg_signal)) / (np.max(bg_signal) - np.min(bg_signal))


        plt.plot(bg_time, bg_subtracted_signal, label = 'subtracted - data')
        plt.plot(bg_time, bg_signal, label = 'bg')
        plt.plot(signal_time, signal, label = 'signal')

        p0 = [0.5, -0.014, 0.001, -0.4]

        fit1, cov1 = op.curve_fit(gaussian, signal_time, signal, p0, maxfev = 1000000)
        plt.plot(signal_time, gaussian(signal_time, *p0), label = 'guess - signal')
        plt.plot(signal_time, gaussian(signal_time, *fit1), label = 'fit - signal')

        fit2, cov2 = op.curve_fit(gaussian, bg_time, bg_signal, p0)
        plt.plot(bg_time, gaussian(bg_time, *fit2), label = 'fit - bg')


        plt.title(angle)
        plt.legend()
        plt.show()

        print(fit1)
        print(fit2)

        diff_in_mean_time = np.abs(fit2[1] - fit1[1])

        #shift signal by difference and plot
        signal_time = signal_time + diff_in_mean_time
        

        plt.plot(bg_time, bg_signal, label = 'bg')
        plt.plot(signal_time, signal, label = 'signal')


        print(signal_time.iloc[0])
        print(bg_time.iloc[-1])
        plt.legend()
        plt.title('shifted - ' + str(angle) + ' degrees')
        plt.show()

        desired_start = signal_time.iloc[0]

        closest_index_start = (bg_time - desired_start).abs().idxmin()
        cropped_bg_time = bg_time.loc[closest_index_start:].reset_index(drop=True)
        cropped_bg_signal = bg_signal.loc[closest_index_start:].reset_index(drop=True)


        desired_end = bg_time.iloc[-1]

        closest_index_end = (signal_time - desired_end).abs().idxmin()
        cropped_signal_time = signal_time.loc[:closest_index_end].reset_index(drop=True)
        cropped_signal = signal.loc[:closest_index_end].reset_index(drop=True)

        plt.plot(cropped_bg_time, cropped_bg_signal, label = 'bg cropped')
        plt.plot(cropped_signal_time, cropped_signal, label = 'signal cropped')

        new_subtracted_signal = cropped_signal - cropped_bg_signal

        plt.plot(cropped_signal_time, new_subtracted_signal, label = 'subtracted signal - new')
        plt.legend()
        plt.title('shifted and cropped - ' + str(angle) + ' degrees')
        plt.show()


    residue_array = residue_df.to_numpy()
    return residue_array


arr = data_get_and_data_go()
