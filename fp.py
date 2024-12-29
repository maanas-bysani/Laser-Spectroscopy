import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.constants as const
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pathlib import Path

plt.rcParams["figure.figsize"] = (10, 6)

# angles = np.arange(0, 200, 60)
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

def n_lorentzians(x, params):
    if len(params) % 4 != 0:
        raise ValueError("Number of parameters must be a multiple of 4 (amp, mean, width, offset for each Lorentzian).")
    result = np.zeros_like(x, dtype=float)
    for amp, mean, width, offset in params:
        result += lorentzian_with_offset(x, amp, mean, width, offset)
    return result


file_path = Path(r"D:\maanas lab work\12 dec")

# file = file_path / f"{angle}.csv"
file = file_path / f"180.csv"

data = pd.read_csv(file)

time = data["in s"]
c2 = data["C2 in V"]
c3 = data["C3 in V"]
c4 = data["C4 in V"]

c2_scaled = (c2 - np.min(c2)) / (np.max(c2) - np.min(c2))
c3_scaled = (c3 - np.min(c3)) / (np.max(c3) - np.min(c3))
c4_scaled = (c4 - np.min(c4)) / (np.max(c4) - np.min(c4))


plt.plot(time, c2_scaled, label = "c2")
plt.plot(time, c3_scaled, label = "c3")
plt.plot(time, c4_scaled, label = "c4")

plt.legend()
plt.show()


def analysis(angles = angles, df = df):
    file_path = Path(r"D:\maanas lab work\12 dec")
    for angle in angles:

        file = file_path / f"{angle}.csv"
        # file = file_path / f"180.csv"

        data = pd.read_csv(file)

        time = df["Reference Time"]
        time = time - time[0]

        fp_data = data["C3 in V"]
        fp_data_scaled = (fp_data - np.min(fp_data)) / (np.max(fp_data) - np.min(fp_data))

        subtracted_signal = df[f"{angle}"]
        plt.plot(time, subtracted_signal, label = 'Subtracted Spectrum')
        plt.plot(time, fp_data_scaled, label = 'Fabry Perot Spectrum')
        plt.legend()
        plt.show()
