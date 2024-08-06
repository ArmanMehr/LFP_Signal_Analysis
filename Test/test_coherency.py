import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def compute_stft(signal, fs, nperseg):
    """Compute the Short-Time Fourier Transform (STFT) of a signal."""
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    return f, t, Zxx

def spectral_coherency(Zxx1, Zxx2):
    """Compute the spectral coherency between two STFT results."""
    # Compute the cross-spectral density
    cross_spectral_density = Zxx1 * np.conj(Zxx2)

    # Compute the power spectral density for each signal
    power1 = np.abs(Zxx1) ** 2
    power2 = np.abs(Zxx2) ** 2

    # Compute spectral coherency
    coherency = (np.abs(cross_spectral_density) ** 2) / (power1 * power2)
    return coherency

# Example usage
if __name__ == "__main__":

    # Parameters
    fs = 256  # Sampling frequency
    t = np.arange(0, 2, 1/fs)  # Time vector

    # Generate synthesized signals
    # Signal 1: Sine wave at 5 Hz with noise
    signal1 = np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(len(t))

    # Signal 2: Sine wave at 5 Hz with a phase shift and noise
    signal2 = np.sin(2 * np.pi * 5 * t + np.pi/4) + 0.05 * np.random.randn(len(t))

    # Compute STFT for both signals
    nperseg = 64  # Length of each segment for STFT
    f1, t1, Zxx1 = compute_stft(signal1, fs, nperseg)
    f2, t2, Zxx2 = compute_stft(signal2, fs, nperseg)

    # Compute spectral coherency
    # coherency = spectral_coherency(Zxx1, Zxx2)
    coherency = (np.abs(Zxx1 * np.conj(Zxx2)) ** 2) / (np.abs(Zxx1) * np.abs(Zxx2))**2

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot Signal 1
    plt.subplot(4, 1, 1)
    plt.plot(t, signal1, label='Signal 1 (5 Hz)', color='blue')
    plt.title('Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    # Plot Signal 2
    plt.subplot(4, 1, 2)
    plt.plot(t, signal2, label='Signal 2 (5 Hz, phase shifted)', color='orange')
    plt.title('Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    # Plot STFT Magnitude of Signal 1
    plt.subplot(4, 1, 3)
    plt.pcolormesh(t1, f1, np.abs(Zxx1), shading='gouraud')
    plt.colorbar(label='Magnitude')
    plt.title('STFT Magnitude of Signal 1')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    # Plot Spectral Coherency
    plt.subplot(4, 1, 4)
    plt.pcolormesh(t1, f1, coherency, shading='gouraud')
    plt.colorbar(label='Spectral Coherency')
    plt.title('Spectral Coherency between Signal 1 and Signal 2')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # plt.ylim([0,10])

    plt.tight_layout()
    plt.show()