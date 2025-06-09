import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, group_delay


# Bandpass filter design
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
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
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
    derivative_signal = np.diff(signal, n=1)  # First-order difference
    return derivative_signal


# Squaring Function
def squaring(signal):
    """
    Squares the ECG signal.

    Parameters:
    - signal: The input ECG signal.

    Returns:
    - squared_signal: The squared ECG signal.
    """
    squared_signal = signal ** 2  # Square each value in the signal
    return squared_signal


# Moving Window Integration Function
def moving_window_integration(signal, window_size=150):
    """
    Performs moving window integration.

    Parameters:
    - signal: The input signal.
    - window_size: The size of the moving window.

    Returns:
    - integrated_signal: The integrated signal.
    """
    window = np.ones(window_size) / window_size
    integrated_signal = np.convolve(signal, window, mode='same')
    return integrated_signal


# Static Thresholding Function (Dual Thresholds)
def static_threshold(signal, upper_factor=1.5, lower_factor=1.0):
    """
    Static thresholding with dual thresholds for QRS detection.

    Parameters:
    - signal: The input ECG signal.
    - upper_factor: Factor for the upper threshold.
    - lower_factor: Factor for the lower threshold.

    Returns:
    - thresholded_signal: The thresholded signal with detected QRS.
    """
    mean_signal = np.mean(signal)
    upper_threshold = mean_signal * upper_factor
    lower_threshold = mean_signal * lower_factor
    thresholded_signal = np.where(signal > upper_threshold, 1, 0)
    thresholded_signal = np.where(signal > lower_threshold, thresholded_signal, 0)
    return thresholded_signal


# LMS-based Adaptive Thresholding Function
def lms_threshold(signal, mu=0.01, window_size=150):
    """
    Applies LMS-based adaptive thresholding to the signal.

    Parameters:
    - signal: The input signal to detect QRS complexes.
    - mu: Step size for the LMS algorithm.
    - window_size: Size of the moving window for calculating the signal.

    Returns:
    - thresholded_signal: The output signal with dynamic thresholds.
    """
    threshold = np.zeros_like(signal)
    error = np.zeros_like(signal)

    # Initialize weight (threshold) with the mean of the first window_size samples
    weight = np.mean(signal[:window_size])

    # Implement LMS update for each sample
    for i in range(window_size, len(signal)):
        # Compute the error as the difference between the signal and the threshold
        error[i] = signal[i] - weight
        # Update the threshold directly based on the error
        weight += mu * error[i]
        # Apply the threshold
        threshold[i] = 1 if signal[i] > weight else 0

    return threshold


# Load ECG signal and annotations from MIT-BIH database
record = wfdb.rdrecord( r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100")  # Replace with your actual path
annotation = wfdb.rdann( r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\100", 'atr')
ecg_signal = record.p_signal
time = np.arange(0, len(ecg_signal)) / record.fs  # Time in seconds

# Apply bandpass filter (5-15 Hz) to the ECG signal
filtered_ecg = bandpass_filter(ecg_signal[:, 0])

# Apply derivative to the filtered ECG signal
derivative_ecg = derivative(filtered_ecg)

# Apply squaring to the derivative signal
squared_signal = squaring(derivative_ecg)

# Apply moving window integration to the squared signal
integrated_signal = moving_window_integration(squared_signal)

# Apply static thresholding
thresholded_signal_static = static_threshold(integrated_signal)

# Apply LMS thresholding
thresholded_signal_lms = lms_threshold(integrated_signal)

# --- Plotting Each Stage Separately ---

# Plot Original ECG Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[:1000], ecg_signal[:1000, 0], color='blue')
plt.title('Original ECG Signal (First Lead)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Filtered ECG Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[:1000], filtered_ecg[:1000], color='red')
plt.title('Filtered ECG Signal (5-15 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Derivative of Filtered ECG Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], derivative_ecg[:1000], color='green')
plt.title('Derivative of Filtered ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Squared Derivative Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], squared_signal[:1000], color='purple')
plt.title('Squared Derivative of ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Integrated Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], integrated_signal[:1000], color='orange')
plt.title('Integrated Signal (Moving Window)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Static Thresholded Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], thresholded_signal_static[:1000], color='black')
plt.title('Static Thresholded Signal (QRS Detection)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot LMS Thresholded Signal (First 1000 samples)
plt.figure(figsize=(10, 6))
plt.plot(time[1:1001], thresholded_signal_lms[:1000], color='green')
plt.title('LMS Adaptive Thresholded Signal (QRS Detection)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


def evaluate_detection(true_annotations, detected_peaks, tolerance=10):
    """
    Evaluates the QRS detection performance using TP, FP, TN, FN, sensitivity, precision, and F1 score.

    Parameters:
    - true_annotations: Actual QRS locations (indices).
    - detected_peaks: Detected QRS locations (indices).
    - tolerance: The tolerance in samples for matching detected peaks with true annotations (e.g., ±10 samples).

    Returns:
    - tp: True positives (detected and correct).
    - fp: False positives (incorrect detections).
    - tn: True negatives (correctly ignored positions).
    - fn: False negatives (missed detections).
    - sensitivity: True positive rate.
    - precision: Positive predictive value.
    - f1_score: Harmonic mean of sensitivity and precision.
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Sort the true annotations and detected peaks for efficient matching
    true_annotations = sorted(true_annotations)
    detected_peaks = sorted(detected_peaks)

    # Pointers for true annotations and detected peaks
    true_idx = 0
    detected_idx = 0

    # Match detected peaks to true annotations within tolerance window
    while true_idx < len(true_annotations) and detected_idx < len(detected_peaks):
        true_peak = true_annotations[true_idx]
        detected_peak = detected_peaks[detected_idx]

        # If the detected peak is within the tolerance range of the true annotation
        if abs(detected_peak - true_peak) <= tolerance:
            tp += 1
            true_idx += 1
            detected_idx += 1
        # If detected peak is too early, it's a false positive
        elif detected_peak < true_peak - tolerance:
            fp += 1
            detected_idx += 1
        # If true annotation is too early, it's a false negative
        else:
            fn += 1
            true_idx += 1

    # Handle remaining true annotations as false negatives
    fn += len(true_annotations) - true_idx
    # Handle remaining detected peaks as false positives
    fp += len(detected_peaks) - detected_idx

    # True negatives are positions correctly ignored by the algorithm,
    # so we are not explicitly calculating TN in this case. It can be inferred
    # from the total number of samples, but it's not needed in typical QRS evals.
    tn = 0

    # Sensitivity, Precision, F1 Score
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return tp, fp, tn, fn, sensitivity, precision, f1_score


# Assuming you have the true annotation peak indices (annotations from MIT-BIH)
true_annotations = annotation.sample  # Annotation peaks

# Detect QRS using static and LMS thresholding
static_peaks = np.where(thresholded_signal_static == 1)[0]
lms_peaks = np.where(thresholded_signal_lms == 1)[0]

# Evaluate the performance for static thresholding with a tolerance of 10 samples
tp_static, fp_static, tn_static, fn_static, sensitivity_static, precision_static, f1_static = evaluate_detection(true_annotations, static_peaks, tolerance=10)

# Evaluate the performance for LMS adaptive thresholding with a tolerance of 10 samples
tp_lms, fp_lms, tn_lms, fn_lms, sensitivity_lms, precision_lms, f1_lms = evaluate_detection(true_annotations, lms_peaks, tolerance=10)

# Print Evaluation Results for Static Thresholding
print("Static Thresholding Performance:")
print(f"TP: {tp_static}, FP: {fp_static}, TN: {tn_static}, FN: {fn_static}")
print(f"Sensitivity: {sensitivity_static:.4f}, Precision: {precision_static:.4f}, F1 Score: {f1_static:.4f}")

# Print Evaluation Results for LMS Adaptive Thresholding
print("\nLMS Adaptive Thresholding Performance:")
print(f"TP: {tp_lms}, FP: {fp_lms}, TN: {tn_lms}, FN: {fn_lms}")
print(f"Sensitivity: {sensitivity_lms:.4f}, Precision: {precision_lms:.4f}, F1 Score: {f1_lms:.4f}")





# # --- Filter Analysis: Magnitude & Phase Response, Pole-Zero Plot, Group Delay ---
#
# # Bandpass filter design for analysis
# def bandpass_filter_design(lowcut=5.0, highcut=15.0, fs=200.0, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
#
#
# # Frequency response (Magnitude and Phase)
# def plot_frequency_response(b, a, fs=200.0, title=""):
#     w, h = freqz(b, a, worN=2000)
#     plt.figure(figsize=(12, 6))
#
#     # Magnitude Response
#     plt.subplot(2, 1, 1)
#     plt.plot(w * fs / (2 * np.pi), abs(h), 'b')
#     plt.title(f'Magnitude Response for {title}')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Amplitude')
#
#     # Phase Response
#     plt.subplot(2, 1, 2)
#     plt.plot(w * fs / (2 * np.pi), np.angle(h), 'b')
#     plt.title(f'Phase Response for {title}')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Phase [radians]')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Plot Pole-Zero Plot with Unit Circle
# def plot_pole_zero(b, a, title=""):
#     z, p, k = tf2zpk(b, a)
#     plt.figure(figsize=(6, 6))
#
#     # Plot unit circle
#     circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
#     plt.gca().add_artist(circle)
#
#     # Plot poles and zeros
#     plt.scatter(np.real(p), np.imag(p), label='Poles', color='red')
#     plt.scatter(np.real(z), np.imag(z), label='Zeros', color='blue')
#     plt.title(f'Pole-Zero Plot for {title}')
#     plt.xlabel('Real')
#     plt.ylabel('Imaginary')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # Group Delay with Safe Handling by Excluding Low Frequencies
# def plot_group_delay(b, a, fs=200.0, title=""):
#     # Calculate group delay for the filter
#     w, gd = group_delay((b, a))
#
#     # Mask to avoid singularities at low frequencies (e.g., exclude the range 0 to 0.05 Hz)
#     mask = (w > 0.05)  # Adjust this threshold as necessary
#     w_filtered = w[mask]
#     gd_filtered = gd[mask]
#
#     # Handle the case when the group delay is singular at certain frequencies
#     gd_filtered = np.nan_to_num(gd_filtered)  # Replace NaN values with 0 (avoid issues with singularities)
#
#     # Plotting group delay for valid frequencies only
#     plt.figure(figsize=(6, 6))
#     plt.plot(w_filtered * fs / (2 * np.pi), gd_filtered, 'b')
#     plt.title(f'Group Delay for {title}')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Group Delay [samples]')
#     plt.grid(True)
#     plt.show()
#
#
# # --- Derivative Filter Design ---
# def derivative_filter_design():
#     # Approximate derivative using a 5-point filter
#     b = np.array([-1, 0, 0, 0, 1])  # Coefficients for the 5-point derivative filter
#     a = np.array([1, 0, 0, 0, 0])   # Denominator for the derivative filter (simple FIR)
#     return b, a
#
#
# # --- Main Execution for Filter Analysis ---
#
# # Bandpass Filter Design for analysis
# b, a = bandpass_filter_design()
#
# # Plot Frequency Response for Bandpass Filter
# plot_frequency_response(b, a, title="Bandpass Filter (5-15 Hz)")
#
# # Plot Pole-Zero Plot for Bandpass Filter
# plot_pole_zero(b, a, title="Bandpass Filter (5-15 Hz)")
#
# # Plot Group Delay for Bandpass Filter
# plot_group_delay(b, a, title="Bandpass Filter (5-15 Hz)")
#
#
# # Moving Window Integration Filter Design for analysis
# def moving_window_integration_filter(window_size=150):
#     b = np.ones(window_size) / window_size
#     a = np.array([1])
#     return b, a
#
#
# # Moving Window Integration filter coefficients for analysis
# b_mwi, a_mwi = moving_window_integration_filter()
#
# # Plot Frequency Response for Moving Window Integration Filter
# plot_frequency_response(b_mwi, a_mwi, title="Moving Window Integration Filter")
#
# # Plot Pole-Zero Plot for Moving Window Integration Filter
# plot_pole_zero(b_mwi, a_mwi, title="Moving Window Integration Filter")
#
# # Plot Group Delay for Moving Window Integration Filter
# plot_group_delay(b_mwi, a_mwi, title="Moving Window Integration Filter")
#
#
# # --- Derivative Filter Analysis ---
#
# # Derivative Filter Design for analysis
# b_deriv, a_deriv = derivative_filter_design()
#
# # Plot Frequency Response for Derivative Filter
# plot_frequency_response(b_deriv, a_deriv, title="Derivative Filter")
#
# # Plot Pole-Zero Plot for Derivative Filter
# plot_pole_zero(b_deriv, a_deriv, title="Derivative Filter")
#
# # Plot Group Delay for Derivative Filter
# plot_group_delay(b_deriv, a_deriv, title="Derivative Filter")
