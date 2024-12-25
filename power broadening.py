import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op
from pathlib import Path
from scipy.odr import *

plt.rcParams["figure.figsize"] = (10,6)

# Parameters
angles = np.arange(0, 361, 60)
# angles = [50]

def data_get_and_data_go(angles = angles):
    # 351 calibration (stool)
    # power_file = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\11 dec data\power data.csv"
    bg_file = r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\11 dec data\BG.csv"
    bg_df = pd.read_csv(bg_file)
    bg_signal = bg_df["C2 in V"]
    time = bg_df["in s"]

    file_path = Path(r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\11 dec data")

    df = pd.DataFrame()
    residue_df = pd.DataFrame()


    for angle in angles:
        file = file_path/f"{angle}.csv"
        print(file)

        data = pd.read_csv(file)
        signal = data["C2 in V"]
        df[f"{angle}"] = data["C2 in V"]

        bg_subtracted_signal = signal - bg_signal

        residue_df[f"{angle}"] = bg_subtracted_signal

        diff_scaled = (bg_subtracted_signal - np.min(bg_subtracted_signal)) / (np.max(bg_subtracted_signal) - np.min(bg_subtracted_signal))
        sig_scaled = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        bg_scaled = (bg_signal - np.min(bg_signal)) / (np.max(bg_signal) - np.min(bg_signal))


        # plt.plot(diff_scaled, label = 'subtracted')
        # plt.plot(bg_scaled, label = 'bg')
        # # plt.plot(time+.00003, sig_scaled, label = 'signal')
        # plt.plot(sig_scaled, label = 'signal')

        plt.plot(bg_subtracted_signal, label = 'subtracted')
        plt.plot(bg_signal, label = 'bg')
        # plt.plot(time+.00003, sig_scaled, label = 'signal')
        plt.plot(signal, label = 'signal')

        plt.title(angle)
        plt.legend()
        plt.show()

    residue_array = residue_df.to_numpy()
    return residue_array


arr = data_get_and_data_go()
# np.savetxt("residue.txt", arr)


print(arr)

plt.plot(arr, label = angles, lw = .2)
plt.legend()
plt.show()
