import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


# 1. Bandpass Filter (5-15 Hz)
def bandpass_filter(signal, fs=200, lowcut=5.0, highcut=15.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    plot_signal(filtered_signal, "Bandpass Filtered Signal (5-15 Hz)", fs)
    return filtered_signal


# 2. Derivative Filter
def derivative(signal):
    diff_signal = np.diff(signal)
    # Pad to maintain length
    diff_signal = np.concatenate(([diff_signal[0]], diff_signal))
    plot_signal(diff_signal, "Derivative of Signal", len(signal))
    return diff_signal


# 3. Squaring Function
def squaring(signal):
    squared_signal = signal ** 2
    plot_signal(squared_signal, "Squared Signal", len(signal))
    return squared_signal


# 4. Moving Window Integration
def moving_window_integration(signal, window_size=30):
    window = np.ones(window_size) / window_size
    integrated_signal = np.convolve(signal, window, mode='same')
    plot_signal(integrated_signal, "Integrated Signal (Moving Window)", len(signal))
    return integrated_signal


# 5. Static Thresholding (modified to return peaks)
def static_threshold(signal, fs=200):
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    threshold = mean_signal + 0.5 * std_signal  # Adjust multiplier as needed

    # Find peaks that cross the threshold
    peaks, _ = find_peaks(signal, height=threshold, distance=int(0.2 * fs))  # 200ms refractory

    plot_signal(signal, "Static Thresholded Signal", fs, peaks=peaks)
    return peaks


# 6. LMS-based Adaptive Thresholding (improved)
def lms_threshold(signal, fs=200, mu=0.01, window_size=150):
    """
    Improved LMS-based adaptive thresholding that tracks signal envelope
    and returns peak locations.
    """
    threshold = np.zeros_like(signal)
    error = np.zeros_like(signal)

    # Initialize with running average of first window
    weight = np.mean(signal[:window_size])

    # Smoothing factor for threshold
    alpha = 0.1

    for i in range(len(signal)):
        # Update threshold more slowly than the error
        if i >= window_size:
            # Use max of recent window for better envelope tracking
            window_max = np.max(signal[i - window_size:i])
            error[i] = window_max - weight
            weight += mu * error[i]

        # Apply smoothing to threshold
        threshold[i] = alpha * weight + (1 - alpha) * threshold[i - 1] if i > 0 else weight

    # Find peaks above the adaptive threshold
    peaks, _ = find_peaks(signal, height=threshold, distance=int(0.2 * fs))

    plot_signal(signal, "LMS Adaptive Thresholded Signal", fs, peaks=peaks, threshold=threshold)
    return peaks


# 7. Post-processing (to remove duplicates and apply refractory period)
def post_process(qrs_peaks, signal, fs=200):
    if len(qrs_peaks) < 2:
        return qrs_peaks

    # Remove duplicates and sort
    qrs_peaks = np.unique(qrs_peaks)

    # Apply refractory period
    min_rr = int(0.2 * fs)  # 200ms
    filtered_peaks = [qrs_peaks[0]]

    for i in range(1, len(qrs_peaks)):
        if qrs_peaks[i] - filtered_peaks[-1] >= min_rr:
            filtered_peaks.append(qrs_peaks[i])

    return np.array(filtered_peaks)


# Update the detect_qrs function to use the corrected thresholding
def detect_qrs(ecg_signal, fs=200, use_lms=True):
    # 1. Bandpass filter
    filtered = bandpass_filter(ecg_signal, fs)

    # 2. Derivative
    diff_signal = derivative(filtered)

    # 3. Squaring
    squared = squaring(diff_signal)

    # 4. Moving window integration
    integrated = moving_window_integration(squared)

    # 5. Thresholding (choose between LMS or Static)
    if use_lms:
        # Use LMS-based adaptive thresholding
        detected_peaks = lms_threshold(integrated, fs)
        method = 'LMS'
    else:
        # Use Static thresholding
        detected_peaks = static_threshold(integrated, fs)
        method = 'Static'

    print(f"Method: {method} Thresholding - Detected {len(detected_peaks)} peaks")
    return detected_peaks



# Evaluation Function
def evaluate_performance(true_peaks, detected_peaks, tolerance=0.1, fs=200):
    tol_samples = int(tolerance * fs)
    tp = 0
    fp = 0
    fn = 0

    matched_true = []
    matched_detected = []

    for true_p in true_peaks:
        found = False
        for det_p in detected_peaks:
            if abs(det_p - true_p) <= tol_samples:
                tp += 1
                matched_true.append(true_p)
                matched_detected.append(det_p)
                found = True
                break
        if not found:
            fn += 1

    fp = len(detected_peaks) - len(matched_detected)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'F1': f1
    }


# Updated plotting function
def plot_signal(signal, title, fs, peaks=None, threshold=None):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(signal)) / fs, signal, label='Signal')

    if threshold is not None:
        plt.plot(np.arange(0, len(threshold)) / fs, threshold, 'r-', label='Threshold', alpha=0.7)

    if peaks is not None:
        plt.plot(peaks / fs, signal[peaks], 'rx', label='Detected Peaks')

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Load MIT-BIH record
    record = wfdb.rdrecord(r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\215", sampto=3000)
    annotation = wfdb.rdann(r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\215", 'atr', sampto=3000)

    ecg_signal = record.p_signal[:, 0]
    true_peaks = annotation.sample

    # Detect QRS complexes using LMS-based adaptive thresholding
    detected_peaks_lms = detect_qrs(ecg_signal, use_lms=True)  # LMS Thresholding

    # Detect QRS complexes using Static thresholding
    detected_peaks_static = detect_qrs(ecg_signal, use_lms=False)  # Static Thresholding

    # Evaluate performance of LMS and Static thresholding
    results_lms = evaluate_performance(true_peaks, detected_peaks_lms)
    results_static = evaluate_performance(true_peaks, detected_peaks_static)

    # Print Results
    print("LMS Adaptive Thresholding Results:")
    print(f"TP: {results_lms['TP']}, FP: {results_lms['FP']}, FN: {results_lms['FN']}")
    print(f"Sensitivity: {results_lms['Sensitivity']:.4f}")
    print(f"Precision: {results_lms['Precision']:.4f}")
    print(f"F1 Score: {results_lms['F1']:.4f}")

    print("\nStatic Thresholding Results:")
    print(f"TP: {results_static['TP']}, FP: {results_static['FP']}, FN: {results_static['FN']}")
    print(f"Sensitivity: {results_static['Sensitivity']:.4f}")
    print(f"Precision: {results_static['Precision']:.4f}")
    print(f"F1 Score: {results_static['F1']:.4f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(ecg_signal)
    plt.plot(true_peaks, ecg_signal[true_peaks], 'go', label='True QRS')
    plt.plot(np.where(detected_peaks_lms == 1)[0], ecg_signal[np.where(detected_peaks_lms == 1)[0]], 'rx', label='Detected QRS (LMS)')
    plt.plot(np.where(detected_peaks_static == 1)[0], ecg_signal[np.where(detected_peaks_static == 1)[0]], 'bx', label='Detected QRS (Static)')
    plt.legend()
    plt.title('QRS Detection Results')
    plt.show()







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