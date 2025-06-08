import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Function to design a bandpass filter
def bandpass_filter(signal, lowcut=5.0, highcut=15.0, fs=200.0, order=4):
    """
    Applies a bandpass filter to the input signal.

    Parameters:
    - signal: Input ECG signal to be filtered.
    - lowcut: Low cutoff frequency (in Hz).
    - highcut: High cutoff frequency (in Hz).
    - fs: Sampling frequency (in Hz).
    - order: The order of the filter (higher order means sharper cutoff).

    Returns:
    - filtered_signal: The ECG signal after bandpass filtering.
    """
    # Calculate the Nyquist frequency (half of the sampling rate)
    nyquist = 0.5 * fs
    
    # Normalize the cutoff frequencies by the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to the signal using filtfilt (zero-phase filtering)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

# Derivative Function
def derivative(signal):
    """
    Computes the derivative (slope) of the ECG signal using a simple 5-point derivative filter.
    
    Parameters:
    - signal: The input ECG signal.

    Returns:
    - derivative_signal: The derivative of the ECG signal.
    """
    derivative_signal = np.diff(signal, n=1)  # Calculate the difference between consecutive points
    return derivative_signal

# Squaring Function (used to square the derivative signal)
def squaring(signal):
    """
    Applies squaring to the input signal to emphasize higher frequencies.
    
    Parameters:
    - signal: The input signal (usually the derivative).

    Returns:
    - squared_signal: The squared version of the input signal.
    """
    squared_signal = signal ** 2  # Square each value in the signal
    return squared_signal

# Moving Window Integration Function (applies moving average to signal)
def moving_window_integration(signal, window_size=150):
    """
    Applies a moving window integration (moving average) to the input signal.
    
    Parameters:
    - signal: Input signal to integrate (usually squared ECG signal).
    - window_size: Size of the integration window (default is 150 samples).
    
    Returns:
    - integrated_signal: The integrated signal after applying the moving window.
    """
    window = np.ones(window_size) / window_size  # Simple moving average window
    integrated_signal = np.convolve(signal, window, mode='same')  # Convolution with the signal
    return integrated_signal

# Dynamic Thresholding Function (used for detecting QRS complexes)
def dynamic_threshold(signal, factor=1.5):
    """
    Apply a dynamic threshold to the input signal based on the mean of the signal.
    
    Parameters:
    - signal: The input signal to threshold.
    - factor: The factor to multiply the mean value to set the threshold (default is 1.5).
    
    Returns:
    - thresholded_signal: A binary signal indicating detected QRS complexes (1) and non-QRS (0).
    """
    threshold = np.mean(signal) * factor  # Set threshold dynamically based on the mean of the signal
    return np.where(signal > threshold, 1, 0)  # Mark points above threshold as 1 (QRS detected)

# Load ECG signal and annotations for Record 100 from MIT-BIH Arrhythmia Database
record = wfdb.rdrecord(r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100")  
annotation = wfdb.rdann(r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100", 'atr')

# Access ECG signal data and annotation information
ecg_signal = record.p_signal  # ECG signal data

# Compute the time axis based on the sampling frequency (fs)
time = np.arange(0, len(ecg_signal)) / record.fs  # Time in seconds

# Apply bandpass filter (5-15 Hz) to the ECG signal (using the first lead if multi-lead ECG)
filtered_ecg = bandpass_filter(ecg_signal[:, 0])  # Using first lead of the signal

# Apply derivative to the filtered ECG signal
derivative_ecg = derivative(filtered_ecg)

# Apply squaring to the derivative signal
squared_signal = squaring(derivative_ecg)

# Apply moving window integration to the squared signal
integrated_signal = moving_window_integration(squared_signal)

# Apply dynamic thresholding to the integrated signal to detect QRS complexes
thresholded_signal = dynamic_threshold(integrated_signal)

# --- Plotting Each Stage Separately ---

# Plot Original ECG Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[:1000], ecg_signal[:1000, 0], color='blue')  # Plot original ECG (First Lead)
plt.title('Original ECG Signal (First Lead)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Filtered ECG Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[:1000], filtered_ecg[:1000], color='red')  # Plot filtered ECG (5-15 Hz bandpass)
plt.title('Filtered ECG Signal (5-15 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Derivative of Filtered ECG Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], derivative_ecg[:1000], color='green')  # Plot derivative (first 1000 samples)
plt.title('Derivative of Filtered ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Squared Derivative Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], squared_signal[:1000], color='purple')  # Plot squared derivative
plt.title('Squared Derivative of ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Integrated Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], integrated_signal[:1000], color='orange')  # Plot integrated signal (moving window)
plt.title('Integrated Signal (Moving Window)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Thresholded Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], thresholded_signal[:1000], color='black')  # Plot thresholded signal (QRS detection)
plt.title('Thresholded Signal (QRS Detection)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
