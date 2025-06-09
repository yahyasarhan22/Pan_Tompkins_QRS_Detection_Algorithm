# Pan-Tompkins QRS Detection Algorithm Enhancement

## Project Overview

This project aims to reproduce and enhance the Pan-Tompkins QRS Detection Algorithm, which is crucial for detecting the QRS complexes in ECG signals. The goal is to implement the original algorithm and then improve it by applying adaptive thresholding using LMS (Least Mean Squares) filtering, ensuring more robust performance in noisy environments.

## Objectives

- **Reproduce the Pan-Tompkins Algorithm**: Recreate the classic QRS detection method as outlined in the paper by Pan & Tompkins (1985), which involves processing ECG signals to detect QRS complexes.
- **Analyze Signal Processing Stages**: Using Digital Signal Processing (DSP) tools, analyze each stage of the signal processing chain, such as frequency response, pole-zero plots, and group delay.
- **Implement Adaptive Thresholding**: Enhance the QRS detection with an adaptive thresholding strategy using LMS filtering to improve robustness, particularly in noisy environments.
- **Evaluate Performance**: Test the performance of both the original and enhanced algorithms using real ECG data from the MIT-BIH Arrhythmia Database.

## Key Tasks

### 1. Literature Review & Signal Flow
- Study the stages of the Pan-Tompkins algorithm: Bandpass filter, Derivative, Squaring, Moving Window Integration, and Thresholding Logic.
- Diagram the entire signal processing flow from the input ECG signal to QRS detection.

### 2. Reproduce the Algorithm
- Implement each stage from the paper using real ECG data (MIT-BIH record 100), detecting and annotating the QRS complexes.

### 3. DSP Filter Analysis
For each filtering stage, analyze and plot the following:
- Magnitude and phase response of the filters.
- Pole-zero plots to understand filter characteristics.
- Group delay to assess time shifts and delays in the system.

### 4. Adaptive Thresholding Using LMS
- Implement an LMS-based adaptive threshold that dynamically adjusts the detection boundary for QRS detection.
- Compare the performance of the original static threshold with the adaptive LMS threshold.

### 5. Evaluation
Compare detection performance based on the following metrics:
- **Sensitivity**: How accurately the algorithm detects QRS complexes.
- **Positive Predictive Value (PPV)**: The proportion of detected complexes that are true positives.
- **F1 Score**: A balance between precision and recall.
- Evaluate performance on both clean and noisy ECG segments.
- Compare the original Pan-Tompkins approach with the LMS-enhanced version.

## Reporting

- Document the entire process, including your code, results (graphs, performance), and findings in a paper-like format.
- Discuss the strengths, limitations, and possible improvements of the algorithm.

## Tools and Libraries

- **Programming Language**: Python or MATLAB.
- **Libraries**:
  - Python: NumPy, SciPy, Matplotlib (for plotting), WFDB (for working with PhysioNet data).
  - MATLAB: Signal Processing Toolbox.
- **Dataset**: MIT-BIH Arrhythmia Database from PhysioNet.

## Expected Learning Outcomes

- **In-depth Understanding**: Gain knowledge about QRS detection and ECG signal processing.
- **DSP Experience**: Hands-on practice with DSP techniques, including filter design and analysis.
- **Practical LMS Application**: Learn to implement adaptive filtering in real-time biomedical applications.
- **Real-Time Signal Processing**: Improve your ability to design and enhance real-time systems for processing ECG signals.

## Additional Extensions (Optional)

- Explore advanced adaptive methods like RLS (Recursive Least Squares), Kalman filters, or machine learning-based thresholding to further improve the algorithm's accuracy and robustness.
- Attempt to write a publishable paper or poster for a conference or journal related to the project.

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# DSP_Projet
# DSP_Projet
# DSP_Projet
