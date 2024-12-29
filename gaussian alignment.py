import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def align_gaussians(file_list, bg_file, output_files=None, plot=False):
    """
    Align multiple Gaussians within each dataset across multiple CSV files so that their peaks overlap.
    Also aligns background data to the signal data.

    Parameters:
        file_list (list of str): List of file paths to CSV files containing the signal data.
                                Each file should have two columns: x and y.
        bg_file (str): File path to the background data CSV file.
                       It should have at least one column "C2 in V".
        output_files (list of str, optional): List of output file paths for the aligned signal data.
                                              If None, aligned data will not be saved.
        plot (bool, optional): If True, plots the original and aligned data for verification.

    Returns:
        aligned_data (list of pd.DataFrame): List of DataFrames with aligned signal data.
        bg_df (pd.DataFrame): Aligned background DataFrame.
    """
    # Load background data
    bg_df = pd.read_csv(bg_file)
    # bg_df.columns = ["in s", "C2 in V"] 
    
    data_frames = []
    peak_positions_list = []

    # Load the signal data and find the x-coordinates of the peaks for each file
    for file in file_list:
        data = pd.read_csv(file)
        # data.columns = ["in s", "C2 in V"]  # Assuming columns are x and y

        # Detect peaks
        peaks, _ = find_peaks(data["C2 in V"], height=0)
        peak_x_positions = data["in s"].iloc[peaks].values

        if len(peak_x_positions) != 4:
            raise ValueError(f"File {file} does not have exactly 4 peaks. Detected peaks: {len(peak_x_positions)}")

        peak_positions_list.append(peak_x_positions)
        data_frames.append(data)

    # Determine the target peak positions (mean of all peak positions for each Gaussian)
    peak_positions_array = np.array(peak_positions_list)
    target_peaks = np.mean(peak_positions_array, axis=0)

    aligned_data = []

    for i, data in enumerate(data_frames):
        # Compute the consistent shift for this dataset
        shift = target_peaks - peak_positions_list[i]
        average_shift = np.mean(shift)

        # Apply the shift
        data["in s"] += average_shift
        aligned_data.append(data)

        if output_files:
            data.to_csv(output_files[i], index=False)

    # Align the background data
    bg_peaks, _ = find_peaks(bg_df["C2 in V"], height=0, distance = 1000)
    bg_peak_x_positions = bg_df["in s"].iloc[bg_peaks].values

    if bg_peak_x_positions is not None:
        bg_shift = target_peaks[0] - bg_peak_x_positions[0]  # Align based on the first peak
        bg_df["in s"] += bg_shift

    if plot:
        plt.figure(figsize=(10, 6))

        # Plot original signal data
        for i, data in enumerate(data_frames):
            plt.plot(data["in s"] - average_shift, data["C2 in V"], '--', label=f'Original Signal File {i+1}')

        # Plot aligned signal data
        for i, data in enumerate(aligned_data):
            plt.plot(data["in s"], data["C2 in V"], label=f'Aligned Signal File {i+1}')

        # Plot aligned background data
        if "in s" in bg_df.columns:
            plt.plot(bg_df["in s"], bg_df["C2 in V"], label='Aligned Background', linestyle='dotted', color='black')

        plt.xlabel("s")
        plt.ylabel("voltage")
        plt.title('Original and Aligned Data')
        plt.legend()
        plt.grid()
        plt.show()

    return aligned_data, bg_df

# Example usage:
file_list = ['50.csv', '150.csv', '200.csv']
bg_file = 'BG.csv'
output_files = ['aligned1.csv', 'aligned2.csv', 'aligned3.csv']
aligned_data, aligned_bg = align_gaussians(file_list, bg_file, output_files=output_files, plot=True)
