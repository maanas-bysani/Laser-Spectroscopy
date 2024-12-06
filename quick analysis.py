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

def std_dev_to_fwhm(std_devs):
    conversion_factor = 2 * np.sqrt(2 * np.log(2))
    fwhms = conversion_factor * np.array(std_devs)
    return fwhms



def analysis():
    data = pd.read_csv("C:\\Users\Maanas\\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\\3 dec.csv")
    # data = pd.read_csv(r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 5 dec\ALL.csv")
    # print(data)

    index = data.index
    time = data.iloc[:,0] - data.iloc[:,0][0]
    channel_1 = data.iloc[:,1]
    channel_2 = data.iloc[:,2]
    channel_3 = data.iloc[:,3]
    ttl_signal = data.iloc[:,-1]


    plt.plot(time, channel_1, label = 'Channel 1')
    plt.plot(time, channel_2, label = 'Channel 2')
    plt.plot(time, channel_3, label = 'Channel 3')
    plt.plot(time, ttl_signal, label = 'TTL Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()

    plt.tight_layout()
    plt.show()


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

    p0 = [-0.400, 0.002, 0.001, 0.00, 
          -1.005, 0.008, 0.001, 0.10, 
          -0.343, 0.020, 0.001, 0.10, 
          -0.40, 0.0305, 0.0015, 0.70          
        ] # 28 nov data - doppler broadened 1

    # p0 = [-0.40, 0.0305, 0.0015, 0.70] 



    fit, cov = op.curve_fit(four_gaussians, time, channel_2, p0)

    print("The parameters")
    print(fit)
    print('--'*45)
    print('The covariance matrix')
    print(cov)


    plt.plot(time, four_gaussians(time, *p0), color='red', label='Guess')
    plt.plot(time, four_gaussians(time, *fit), color='black', label='Fit')
    plt.plot(time, channel_2, color='purple', label='Data - Channel 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("amplitudes :", p0[0], p0[4], p0[8], p0[12])
    print("means :", p0[1], p0[5], p0[9], p0[13])
    print("std dev :", p0[2], p0[6], p0[10], p0[14])
    print("fwhm :", std_dev_to_fwhm([p0[2], p0[6], p0[10], p0[14]]))
    print("offset :", p0[3], p0[7], p0[11], p0[15])



analysis()

# calcualtions

k = const.k
c = const.c
h = const.h
hbar = const.hbar

temp = 300
m = const.m_p * 87

freq = 384.2304844685e12

fwhm = np.sqrt((8 * k * temp * np.log(2))/ (m * c * c)) * freq

print(fwhm * 1e-9)

