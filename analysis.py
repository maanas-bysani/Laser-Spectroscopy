import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.constants as const

def gaussian(x, amp, mu, sigma, c):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c

def four_gaussians(x, amp1, mu1, sigma1, c1, amp2, mu2, sigma2, c2, amp3, mu3, sigma3, c3, amp4, mu4, sigma4, c4):
    return (
        amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) + c1 +
        amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) + c2 +
        amp3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2)) + c3 +
        amp4 * np.exp(-((x - mu4) ** 2) / (2 * sigma4 ** 2)) + c4
    )


def n_gaussians(x, *params):
    if len(params) % 4 != 0:
        raise ValueError("Number of parameters must be a multiple of 4 (amp, mu, sigma, c for each Gaussian).")
    
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        amp, mu, sigma, c = params[i:i+4]
        result += amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
    
    return result


def std_dev_to_fwhm(std_devs):
    conversion_factor = 2 * np.sqrt(2 * np.log(2))
    fwhms = conversion_factor * np.array(std_devs)
    return fwhms



def analysis():
    data = pd.read_csv("C:\\Users\Maanas\\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\\3 dec.csv")
    # print(data)

    index = data.index
    data.iloc[:,0] = data.iloc[:,0] - data.iloc[:,0][0]
    # time = data.iloc[:,0] - data.iloc[:,0][0]
    time = data.iloc[:,0]
    channel_1 = data.iloc[:,1]
    channel_2 = data.iloc[:,2]
    channel_3 = data.iloc[:,3]
    ttl_signal = data.iloc[:,-1]


    plt.plot(time, channel_1, label = 'Channel 1')
    plt.plot(time, channel_2, label = 'Channel 2')
    plt.plot(time, ttl_signal, label = 'TTL Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()

    plt.tight_layout()
    # plt.show()

    # peak_1_start = (np.abs(time)).idxmin()
    peak_1_end = (np.abs(time - 0.006)).idxmin()    
    peak_1_data = data.loc[:peak_1_end]

    peak_2_start = peak_1_end+1
    peak_2_end = (np.abs(time - 0.013)).idxmin()
    peak_2_data = data.loc[peak_2_start:peak_2_end]


    # peak_3_start = (np.abs(time - 0.019)).idxmin()
    peak_3_start = peak_2_end+1
    peak_3_end = (np.abs(time - 0.025)).idxmin()
    peak_3_data = data.loc[peak_3_start:peak_3_end]

    # peak_4_start = (np.abs(time - 0.031)).idxmin()
    peak_4_start = peak_3_end+1
    # peak_4_end = (np.abs(time)).idxmax()
    peak_4_data = data.loc[peak_4_start:]

    chosen_channel = 2
    plt.plot(peak_1_data.iloc[:,0], peak_1_data.iloc[:,chosen_channel], label = 'Dataset 1')
    plt.plot(peak_2_data.iloc[:,0], peak_2_data.iloc[:,chosen_channel], label = 'Dataset 2')
    plt.plot(peak_3_data.iloc[:,0], peak_3_data.iloc[:,chosen_channel], label = 'Dataset 3')
    plt.plot(peak_4_data.iloc[:,0], peak_4_data.iloc[:,chosen_channel], label = 'Dataset 4')
    plt.title('channel {0}' .format(chosen_channel))
    plt.legend()
    # plt.show()


    ## normalisation    
    peak_1_n_data = pd.DataFrame(peak_1_data.iloc[:, 0])
    print(peak_1_data)
    print(time)
    peak_2_n_data = pd.DataFrame(peak_2_data.iloc[:, 0])
    peak_3_n_data = pd.DataFrame(peak_3_data.iloc[:, 0])
    peak_4_n_data = pd.DataFrame(peak_4_data.iloc[:, 0])

    datasets = [peak_1_data, peak_2_data, peak_3_data, peak_4_data]
    datasets_normalised = [peak_1_n_data, peak_2_n_data, peak_3_n_data, peak_4_n_data]

    ## min max normalisation
    # for df, df_normalised in zip(datasets, datasets_normalised):
    #     df_normalised['ch1_n'] = (df.iloc[:,1] - df.iloc[:,1].min()) / (df.iloc[:,1].max() - df.iloc[:,1].min())
    #     df_normalised['ch2_n'] = (df.iloc[:,2] - df.iloc[:,2].min()) / (df.iloc[:,2].max() - df.iloc[:,2].min())
    #     df_normalised['ch3_n'] = (df.iloc[:,3] - df.iloc[:,3].min()) / (df.iloc[:,3].max() - df.iloc[:,3].min())
    #     df_normalised['ttl_n'] = (df.iloc[:,-1] - df.iloc[:,-1].min()) / (df.iloc[:,-1].max() - df.iloc[:,-1].min())


    for df, df_normalised in zip(datasets, datasets_normalised):
        min_peak_1 = df.iloc[:, 1].min()
        min_peak_2 = df.iloc[:, 2].min()

        ratio = min_peak_2 / min_peak_1

        df_normalised['ch1_n'] = df.iloc[:, 1] * ratio
        df_normalised['ch2_n'] = df.iloc[:, 2]
        
    print(peak_1_n_data.columns)
    peak_1_data_shift = 0.0002
    peak_2_data_shift = 0.0001
    peak_3_data_shift = 0.00000
    peak_4_data_shift = 0.00000


    peak_1_data_shift_ch2 = 0
    peak_2_data_shift_ch2 = 0
    peak_3_data_shift_ch2 = 0
    peak_4_data_shift_ch2 = 0.1
    
    plt.plot(peak_1_n_data.iloc[:,0], peak_1_n_data.iloc[:,1], label = 'ch1_n p1')
    plt.plot(peak_1_n_data.iloc[:,0]-peak_1_data_shift, peak_1_n_data.iloc[:,2]-peak_1_data_shift_ch2, label = 'ch2_n p1')

    plt.plot(peak_2_n_data.iloc[:,0], peak_2_n_data.iloc[:,1], label = 'ch1_n p2')
    plt.plot(peak_2_n_data.iloc[:,0]-peak_2_data_shift, peak_2_n_data.iloc[:,2]-peak_2_data_shift_ch2, label = 'ch2_n p2')

    plt.plot(peak_3_n_data.iloc[:,0], peak_3_n_data.iloc[:,1], label = 'ch1_n p3')
    plt.plot(peak_3_n_data.iloc[:,0]-peak_3_data_shift, peak_3_n_data.iloc[:,2]-peak_3_data_shift_ch2, label = 'ch2_n p3')

    plt.plot(peak_4_n_data.iloc[:,0], peak_4_n_data.iloc[:,1], label = 'ch1_n p4')
    plt.plot(peak_4_n_data.iloc[:,0]-peak_4_data_shift, peak_4_n_data.iloc[:,2]-peak_4_data_shift_ch2, label = 'ch2_n p4')

    plt.legend()
    # plt.show()


    peak_1_n_subtracted = (peak_1_n_data.iloc[:,2]-peak_1_data_shift_ch2) - peak_1_n_data.iloc[:,1]
    peak_2_n_subtracted = (peak_2_n_data.iloc[:,2]-peak_2_data_shift_ch2) - peak_2_n_data.iloc[:,1]
    peak_3_n_subtracted = (peak_3_n_data.iloc[:,2]-peak_3_data_shift_ch2) - peak_3_n_data.iloc[:,1]
    peak_4_n_subtracted = (peak_4_n_data.iloc[:,2]-peak_4_data_shift_ch2) - peak_4_n_data.iloc[:,1]


    plt.plot(peak_1_n_data.iloc[:,0], peak_1_n_subtracted, label = 'peak1')
    plt.plot(peak_2_n_data.iloc[:,0], peak_2_n_subtracted, label = 'peak2')
    plt.plot(peak_3_n_data.iloc[:,0], peak_3_n_subtracted, label = 'peak3')
    plt.plot(peak_4_n_data.iloc[:,0], peak_4_n_subtracted, label = 'peak4')
    plt.xlabel('Time')
    plt.ylabel('Intensity (arb.)')
    plt.title('Subtracted Spectrum')
    plt.legend()
    # plt.show()
    subtracted_spectrum = pd.concat([peak_1_n_subtracted, peak_2_n_subtracted, peak_3_n_subtracted, peak_4_n_subtracted], axis=0, ignore_index=True)




    # # normalise
    # channel_1_normalised = (channel_1 - channel_1.min()) / (channel_1.max() - channel_1.min())
    # channel_2_normalised = (channel_2 - channel_2.min()) / (channel_2.max() - channel_2.min())
    # channel_3_normalised = (channel_3 - channel_3.min()) / (channel_3.max() - channel_3.min())
    # ttl_signal_normalised = (ttl_signal - ttl_signal.min()) / (ttl_signal.max() - ttl_signal.min())

    # plt.plot(time, channel_1_normalised, label = 'Channel 1')
    # plt.plot(time, channel_2_normalised, label = 'Channel 2')
    # plt.plot(time,ttl_signal_normalised, label = 'TTL Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Normalised Signal (arb.)')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    
    # gaussian fits

    # x, amp1, mu1, sigma1, amp2, mu2, sigma2, amp3, mu3, sigma3, amp4, mu4, sigma4

    # p0 = [-0.700, 0.01275, 0.001, 0.10, 
    #       -1.571, 0.01843, 0.001, 0.10, 
    #       -0.695, 0.03055, 0.001, 0.20, 
    #       -0.285, 0.04056, 0.001, 0.25
    #     ]

    # p0 = [-0.400, 0.002, 0.001, 0.00, 
    #       -1.005, 0.008, 0.001, 0.10, 
    #       -0.343, 0.020, 0.001, 0.10, 
    #       
    #     ] # 28 nov data - doppler broadened 2

    # p0 = [-0.400, 0.002, 0.001, 0.00, 
    #       -1.005, 0.008, 0.001, 0.10, 
    #       -0.343, 0.020, 0.001, 0.10, 
    #       -0.40, 0.0305, 0.0015, 0.70          
    #     ] # 28 nov data - doppler broadened 1


    p0 = [-0.400, 0.002, 0.001, 0.00, 
          -1.005, 0.008, 0.001, 0.10, 
          -0.343, 0.020, 0.001, 0.10, 
          -0.40, 0.0305, 0.0015, 0.70          
        ] # 3 dec subtracted
    # amp mu sigma c 
    # p0 = [0.09, 0.0013, 0.00005, 0.13]
    # p0 = [0.06, 0.0017, 0.00005, 0.17]
    # p0 = [0.06, 0.00175, 0.00005, 0.17]
    # p0 = [0.50, 0.002, 0.00005, 0.30] # works
    # p0 = [0.80, 0.0025, 0.00005, 0.00] # works
    # p0 = [0.33, 0.00327, 0.000045, -0.10] # somewhat works

    p0 = [0.33, 0.00327, 0.000045, -0.10]

    fit, cov = op.curve_fit(gaussian, time, subtracted_spectrum, p0)

    print("The parameters")
    print(fit)
    print('--'*45)
    print('The covariance matrix')
    print(cov)

    plt.clf()
    plt.plot(time, gaussian(time, *p0), color='red', label='Guess')
    plt.plot(time, gaussian(time, *fit), color='black', label='Fit')
    plt.plot(time, subtracted_spectrum, color='purple', label='Data - Channel 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.tight_layout()
    plt.show()

#     print("amplitudes :", p0[0], p0[4], p0[8], p0[12])
#     print("means :", p0[1], p0[5], p0[9], p0[13])
#     print("std dev :", p0[2], p0[6], p0[10], p0[14])
#     print("fwhm :", std_dev_to_fwhm([p0[2], p0[6], p0[10], p0[14]]))
#     print("offset :", p0[3], p0[7], p0[11], p0[15])


    # # Example: Simulated data for spectra
    # x = time  # Common x-axis
    # orange_spectrum = channel_2  # Simulated orange spectrum
    # blue_spectrum = channel_1  # Simulated blue spectrum

    # # Normalize the spectra
    # orange_normalized = (orange_spectrum - np.min(orange_spectrum)) / (np.max(orange_spectrum) - np.min(orange_spectrum))
    # blue_normalized = (blue_spectrum - np.min(blue_spectrum)) / (np.max(blue_spectrum) - np.min(blue_spectrum))

    # # Subtract the blue spectrum from the orange spectrum
    # subtracted_spectrum = orange_normalized - blue_normalized

    # # Plot the spectra and the result
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, orange_normalized, label="Orange Spectrum (Normalized)", color="orange")
    # plt.plot(x, blue_normalized, label="Blue Spectrum (Normalized)", color="blue")
    # plt.plot(x, subtracted_spectrum, label="Subtracted Spectrum", color="green")
    # plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    # plt.legend()
    # plt.xlabel("X-axis (e.g., wavelength)")
    # plt.ylabel("Normalized Intensity")
    # plt.title("Spectra Subtraction")
    # plt.show()


analysis()




# # calcualtions

# k = const.k
# c = const.c
# h = const.h
# hbar = const.hbar

# temp = 300
# m = const.m_p * 87

# freq = 384.2304844685e12

# fwhm = np.sqrt((8 * k * temp * np.log(2))/ (m * c * c)) * freq

# print(fwhm * 1e-9)

