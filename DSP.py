# Yahya Sarhan                                Fadi Bassous                                                 Maysa Khanfar
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks,freqz,tf2zpk,lfilter,group_delay
import  warnings

# 1. Bandpass Filter (5-15 Hz)
def bandpass_filter(signal):
    """
    Implements the Pan-Tompkins bandpass filter by cascading:
    - Low-pass filter: cutoff ~11 Hz
    - High-pass filter: cutoff ~5 Hz
    Sampling rate = 200 Hz, integer coefficients
    """
    # Low-pass filter: y(n) = 2y(n-1) - y(n-2) + x(n) - 2x(n-6) + x(n-12)
    lp = np.zeros_like(signal)
    for n in range(12, len(signal)):
        lp[n] = 2 * lp[n-1] - lp[n-2] + signal[n] - 2 * signal[n-6] + signal[n-12]

    # High-pass filter: y(n) = y(n-1) - x(n)/32 + x(n-16) - x(n-17) + x(n-32)/32
    hp = np.zeros_like(lp)
    for n in range(32, len(lp)):
        hp[n] = hp[n-1] - lp[n] / 32 + lp[n-16] - lp[n-17] + lp[n-32] / 32

    plot_signal(hp, "Bandpass Filtered Signal", fs=200, time_window_sec=10)
    return hp



# 2. Derivative Filter
def derivative(signal, fs=200):
    """
    Implements the 5-point derivative used in Pan-Tompkins algorithm.
    """
    deriv = np.zeros_like(signal)
    for n in range(2, len(signal)-2):
        deriv[n] = (1/8) * (-signal[n-2] - 2*signal[n-1] + 2*signal[n+1] + signal[n+2])

    plot_signal(deriv, "Derivative of Signal", fs)
    return deriv



# 3. Squaring Function
def squaring(signal):
    squared_signal = signal ** 2
    plot_signal(squared_signal, "Squared Signal", len(signal), time_window_sec=5)
    return squared_signal


# 4. Moving Window Integration
def moving_window_integration(signal, window_size=30):
    # window_size: number of samples in the moving window.
    window = np.ones(window_size) / window_size
    """
        This line performs convolution, which means:
          Slide the averaging window across the signal.
          At each point, multiply-and-sum the overlapping values → get a smoothed version of the signal.
        mode='same' ensures the output signal is the same length as the input.
    """
    integrated_signal = np.convolve(signal, window, mode='same')
    plot_signal(integrated_signal, "Integrated Signal (Moving Window)", len(signal), time_window_sec=10)
    return integrated_signal


# 5. Static Thresholding (modified to return peaks)
def static_threshold(signal, fs=200, plot_only=False):
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    #std stands for standard deviation, it measures how much the values in a signal vary around the mean.
    threshold = mean_signal + 0.5 * std_signal  # Adjust multiplier as needed

    threshold_array = np.full_like(signal, threshold)  # ✅ always define this

    if plot_only:
        # Just plot with the threshold line, no peak detection
        plot_signal(signal, "Static Thresholded Signal (preview only)", fs, threshold=threshold_array, time_window_sec=10)
        return None
    else:
        # Normal detection mode
        # Find peaks above threshold
        peaks, _ = find_peaks(signal, height=threshold, distance=int(0.2 * fs))
        plot_signal(signal, "Static Thresholded Signal (with detected peaks)", fs, peaks=peaks, threshold=threshold_array, time_window_sec=10)
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

    plot_signal(signal, "LMS Adaptive Thresholded Signal", fs, peaks=peaks, threshold=threshold, time_window_sec=10)
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
    filtered = bandpass_filter(ecg_signal)

    # 2. Derivative
    diff_signal = derivative(filtered)

    # 3. Squaring
    squared = squaring(diff_signal)

    # 4. Moving window integration
    integrated = moving_window_integration(squared)

    # ONLY show static threshold for LMS case (for comparison)
    if use_lms:
        static_threshold(integrated, fs, plot_only=True)

    # 5. Thresholding
    if use_lms:
        detected_peaks = lms_threshold(integrated, fs)
        method = 'LMS'
    else:
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
def plot_signal(signal, title, fs, peaks=None, threshold=None, time_window_sec=10, normalize=False, smooth=True):
    plt.figure(figsize=(12, 5))

    # Generate time axis
    time = np.arange(len(signal)) / fs

    # Determine number of samples to display
    max_samples = int(time_window_sec * fs)
    signal = signal[:max_samples]
    time = time[:max_samples]
    if threshold is not None:
        threshold = threshold[:max_samples]
    if peaks is not None:
        peaks = peaks[peaks < max_samples]

    # Optional smoothing for visualization
    if smooth:
        window_size = 5  # small window just for visualization
        kernel = np.ones(window_size) / window_size
        signal = np.convolve(signal, kernel, mode='same')
        if threshold is not None:
            threshold = np.convolve(threshold, kernel, mode='same')

    # Optional normalization
    if normalize:
        min_val = np.min(signal)
        max_val = np.max(signal)
        signal = (signal - min_val) / (max_val - min_val)
        if threshold is not None:
            threshold = (threshold - min_val) / (max_val - min_val)

    # Plot
    plt.plot(time, signal, label='Signal', linewidth=1.2, alpha=0.9)
    if threshold is not None:
        plt.plot(time, threshold, 'r--', label='Threshold', linewidth=1)
    if peaks is not None and len(peaks) > 0:
        plt.plot(peaks / fs, signal[peaks], 'rx', label='Detected Peaks', markersize=6)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()



#################

# Suppress specific warnings we expect and understand
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# 1. Bandpass Filter Analysis (as per original paper)
def original_bandpass_filters(fs=200):
    """Implement the exact filters from the Pan-Tompkins paper"""
    # Low-pass filter (cutoff ~11 Hz)
    # H(z) = (1 - z^-6)^2 / (1 - z^-1)^2
    b_low = np.array([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1])
    a_low = np.array([1, -2, 1])

    # High-pass filter (cutoff ~5 Hz)
    # H(z) = (-1 + 32z^-16 + z^-32) / (1 + z^-1)
    b_high = np.zeros(33)
    b_high[0] = -1
    b_high[16] = 32
    b_high[32] = 1
    a_high = np.array([1, 1])

    return b_low, a_low, b_high, a_high


# 2. Derivative Filter (as per original paper)
def original_derivative_filter(fs=200):
    """5-point derivative from the paper"""
    # H(z) = (1/8T)(-z^-2 - 2z^-1 + 2z^1 + z^2)
    b = np.array([-1, -2, 0, 2, 1]) / (8 * 1 / fs)  # T = 1/fs
    a = np.array([1])
    return b, a


# 3. Moving Window Integration (as per original paper)
def original_integrator(window_size=30):
    """Moving window integrator from the paper"""
    b = np.ones(window_size) / window_size
    a = np.array([1])
    return b, a


# Function to plot frequency response with safe dB calculation
def plot_frequency_response(b, a, fs=200, title=""):
    w, h = freqz(b, a, worN=2000)
    plt.figure(figsize=(12, 6))

    # Magnitude Response (with safe dB calculation)
    magnitude = np.abs(h)
    # Replace zeros with small value to avoid log10(0)
    magnitude[magnitude == 0] = np.finfo(float).eps
    dB = 20 * np.log10(magnitude)

    plt.subplot(2, 1, 1)
    plt.plot(w * fs / (2 * np.pi), dB, 'b')
    plt.title(f'Magnitude Response for {title}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)

    # Phase Response
    plt.subplot(2, 1, 2)
    plt.plot(w * fs / (2 * np.pi), np.angle(h), 'b')
    plt.title(f'Phase Response for {title}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Function to plot pole-zero plot
def plot_pole_zero(b, a, title=""):
    z, p, k = tf2zpk(b, a)
    plt.figure(figsize=(6, 6))

    # Unit circle
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    plt.gca().add_patch(unit_circle)

    # Plot poles and zeros
    plt.scatter(np.real(z), np.imag(z), marker='o', color='b', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')

    plt.title(f'Pole-Zero Plot for {title}')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.show()


# Function to plot group delay with safe handling
def plot_group_delay(b, a, fs=200, title=""):
    try:
        w, gd = group_delay((b, a), w=2000)
        # Replace any inf/nan values with 0
        gd = np.nan_to_num(gd)

        plt.figure(figsize=(10, 5))
        plt.plot(w * fs / (2 * np.pi), gd, 'b')
        plt.title(f'Group Delay for {title}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Group Delay [samples]')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Could not calculate group delay for {title}: {str(e)}")

###########
# Example Usage
if __name__ == "__main__":

    # === Helper Function to Load ECG Record ===
    def load_ecg_record(record_number, sampto=None):
        base_path = r"C:\Users\yahya_k6rln48\OneDrive\Desktop\DSP_Project\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"
        record = wfdb.rdrecord(f"{base_path}\\{record_number}", sampto=sampto)
        annotation = wfdb.rdann(f"{base_path}\\{record_number}", 'atr', sampto=sampto)
        return record, annotation

    DELAY = 39  # Processing delay


    def run_analysis(record_number, label, sampto=11000):
        record, annotation = load_ecg_record(record_number, sampto=sampto)
        ecg_signal = record.p_signal[:, 0]
        fs = 200
        DELAY = 39

        # Delay-corrected true peaks
        true_peaks = annotation.sample - DELAY
        true_peaks = true_peaks[(true_peaks >= 0) & (true_peaks < len(ecg_signal))]

        # Plot original signal
        plot_signal(ecg_signal, f"{label} - Original ECG Signal", fs)

        # Run both detectors
        peaks_lms = detect_qrs(ecg_signal, fs=fs, use_lms=True)
        peaks_static = detect_qrs(ecg_signal, fs=fs, use_lms=False)

        # Delay-correct detected peaks
        peaks_lms = peaks_lms - DELAY
        peaks_static = peaks_static - DELAY

        peaks_lms = peaks_lms[(peaks_lms >= 0) & (peaks_lms < len(ecg_signal))]
        peaks_static = peaks_static[(peaks_static >= 0) & (peaks_static < len(ecg_signal))]

        # Evaluate performance
        eval_lms = evaluate_performance(true_peaks, peaks_lms, fs=fs)
        eval_static = evaluate_performance(true_peaks, peaks_static, fs=fs)

        print(f"\n====== {label.upper()} ECG (Record {record_number}) ======")
        print("Static Thresholding:", eval_static)
        print("LMS Thresholding:   ", eval_lms)

        # Final Plot
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_signal, label="ECG Signal")
        plt.plot(true_peaks, ecg_signal[true_peaks], 'go', label='True QRS')
        plt.plot(peaks_lms, ecg_signal[peaks_lms], 'rx', label='Detected QRS (LMS)')
        plt.plot(peaks_static, ecg_signal[peaks_static], 'bx', label='Detected QRS (Static)')
        plt.title(f'{label} ECG - QRS Detection (LMS vs Static)')
        plt.legend()
        plt.grid(True)
        plt.show()


    # === Run for Clean and Noisy signals ===
    run_analysis("100", "Clean", sampto=None)
    run_analysis("215", "Noisy", sampto=None)

    ########################################
    # Sampling frequency from the paper
    fs = 200
    b_low, a_low, b_high, a_high = original_bandpass_filters(fs)

    print("\nAnalyzing Low-pass Filter (cutoff ~11 Hz)")
    plot_frequency_response(b_low, a_low, fs, "Low-pass Filter")
    plot_pole_zero(b_low, a_low, "Low-pass Filter")
    plot_group_delay(b_low, a_low, fs, "Low-pass Filter")

    print("\nAnalyzing High-pass Filter (cutoff ~5 Hz)")
    plot_frequency_response(b_high, a_high, fs, "High-pass Filter")
    plot_pole_zero(b_high, a_high, "High-pass Filter")
    plot_group_delay(b_high, a_high, fs, "High-pass Filter")

    print("\nAnalyzing Combined Bandpass Filter")
    w, h_low = freqz(b_low, a_low, worN=2000)
    _, h_high = freqz(b_high, a_high, worN=2000)
    h_combined = h_low * h_high
    magnitude = np.abs(h_combined)
    magnitude[magnitude == 0] = np.finfo(float).eps
    dB = 20 * np.log10(magnitude)

    plt.figure(figsize=(12, 6)) 
    plt.plot(w * fs / (2 * np.pi), dB, 'b')
    plt.title('Magnitude Response for Combined Bandpass Filter')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.show()

    b_deriv, a_deriv = original_derivative_filter(fs)
    print("\nAnalyzing Derivative Filter")
    plot_frequency_response(b_deriv, a_deriv, fs, "Derivative Filter")
    plot_pole_zero(b_deriv, a_deriv, "Derivative Filter")
    plot_group_delay(b_deriv, a_deriv, fs, "Derivative Filter")

    b_integ, a_integ = original_integrator()
    print("\nAnalyzing Moving Window Integrator")
    plot_frequency_response(b_integ, a_integ, fs, "Moving Window Integrator")
    plot_pole_zero(b_integ, a_integ, "Moving Window Integrator")
    plot_group_delay(b_integ, a_integ, fs, "Moving Window Integrator")

