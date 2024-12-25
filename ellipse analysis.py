import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def polarization_intensity(theta, A, B, theta_0):
    theta_rad = np.radians(theta)
    return A + B * np.cos(2 * (theta_rad - np.radians(theta_0)))

# data = pd.read_csv(r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\good data - 6 dec\linear polariser data.csv")
data = pd.read_csv(r"C:\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Cycle 3\Data\13 dec\lin pol fixed data.csv")

angles = data["angle"]
powers = data["mean"]

initial_guess = [np.mean(powers), (np.max(powers) - np.min(powers)), 0]
popt, pcov = curve_fit(polarization_intensity, angles, powers, p0=initial_guess)

A, B, theta_0 = popt

P_max = A + B
P_min = A - B
ellipticity = np.sqrt(P_min / P_max)

print(f"Fitted Parameters:")
print(f"  A (Offset): {A:.3f}")
print(f"  B (Amplitude): {B:.3f}")
print(f"  Theta_0 (Major Axis Orientation): {theta_0:.3f} degrees")
print(f"Ellipticity: {ellipticity:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(angles, powers, label="Measured Data", color="tab:blue")
plt.plot(angles, polarization_intensity(angles, *popt), label="Fitted Curve", color="tab:red")
plt.xlabel("Angle (degrees)")
plt.ylabel("Detected Power")
plt.title("Polarization Fit")
plt.legend()
plt.show()

def plot_polarization_ellipse(P_max, P_min, theta_0):
    t = np.linspace(0, 2 * np.pi, 500)
    x = P_max * np.cos(t)
    y = P_min * np.sin(t)

    theta_0_rad = np.radians(theta_0)
    x_rot = x * np.cos(theta_0_rad) - y * np.sin(theta_0_rad)
    y_rot = x * np.sin(theta_0_rad) + y * np.cos(theta_0_rad)

    plt.figure(figsize=(6, 6))
    plt.plot(x_rot, y_rot, label="Polarization Ellipse")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polarization Ellipse")
    plt.axis('equal')
    plt.legend()
    plt.show()

plot_polarization_ellipse(P_max, P_min, theta_0)

plt.polar(angles*np.pi/180, powers)
plt.title("Polar plot")
plt.show()


def calculate_qwp_angle(P_max, P_min, theta_0):
    theta_0_rad = np.radians(theta_0)

    qwp_angle = theta_0 / 2

    print(f"The fast axis of the QWP should be at {qwp_angle:.3f} degrees.")

    return qwp_angle

qwp_angle = calculate_qwp_angle(P_max, P_min, theta_0)


A_x = np.sqrt(P_max)
A_y = np.sqrt(P_min)
delta = 0

E_x = A_x
E_y = A_y * np.exp(1j * delta)

rotation_matrix = np.array([
    [np.cos(theta_0), -np.sin(theta_0)],
    [np.sin(theta_0), np.cos(theta_0)]
])

jones_vector = np.array([E_x, E_y])
rotated_jones_vector = np.dot(rotation_matrix, jones_vector)

jones_matrix = np.outer(jones_vector, np.conjugate(jones_vector))

print("Jones vector:", jones_vector)
print("Jones vector (rotated):", rotated_jones_vector)
print("Jones matrix representation:\n", jones_matrix)
