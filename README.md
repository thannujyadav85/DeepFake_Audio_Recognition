# Deepfake Audio Detection Engine

An end-to-end, lightweight deep learning pipeline designed to distinguish between **Genuine Human Utterances** and **AI-Generated Deepfakes** (Voice Clones). This system processes raw audio signals, extracts time-frequency acoustic characteristics, and runs a custom 2D Convolutional Neural Network (CNN) to achieve high-accuracy real-world forensics.

---

## System Architecture & Workflow

The framework follows a sequential four-stage signal processing and machine learning workflow:
### 1. Data Preprocessing Pipeline
To feed variable-length audio signals into a structured tensor frame without shape mismatches, raw inputs undergo temporal and dimensional normalization:

* **Sampling Rate Uniformity:** Every incoming audio sample is forced to a standardized target of **$16,000 \text{ Hz}$ ($16 \text{ kHz}$)**. This eliminates high-frequency noise artifacts while maintaining the acoustic bandwidth required to detect synthetic vocoder signatures.
* **Strict 3-Second Temporal Mapping:** All audio signals are mapped to an exact duration of **$3.0\text{ seconds}$** ($48,000\text{ total discrete audio samples}$).
  * **Padding Strategy:** Utterances shorter than 3.0 seconds are extended using trailing **constant zero-padding**.
  * **Trimming Strategy:** Utterances longer than 3.0 seconds are sliced down using a strict **head-truncation constraint** ($t_{0}$ to $t_{48000}$).

### 2. Feature Extraction Engine (Mel-Spectrograms)
The 1D time-domain audio waveform is mathematically transformed into a 2D spatial log-frequency matrix using a short-time Fourier pipeline:

1. **Short-Time Fourier Transform (STFT):** Computed using a Hann windowing function with an $N_{\text{fft}}$ window size of $2048$ samples and a hop length of $512$ samples.
2. **Mel-Scale Filterbank:** The linear power spectrum is mapped onto a Mel-scale filterbank using $128$ distinct frequency bins ($N_{\text{mels}} = 128$) to mimic human auditory perception.
3. **Power-to-dB Log Scaling:** The raw acoustic power values are compressed into decibels ($\text{dB}$) via log scaling:
   $$\text{Mel}_{\text{dB}} = 10 \cdot \log_{10}\left(\frac{\text{Mel}_{\text{power}}}{\max(\text{Mel}_{\text{power}})}\right)$$
4. **Min-Max Normalization:** To prevent gradient saturation across deep network layers, the decibel matrix is normalized strictly between **$[-1, 1]$**.

**Final Feature Input Dimensions:** $\mathbf{X} \in \mathbb{R}^{\text{Batch} \times \text{Channel} \times \text{Freq} \times \text{Time}} \longrightarrow [B \times 1 \times 128 \times 94]$

---

## Deep Learning Model Architecture

The core classifier is a custom, deep 2D Convolutional Neural Network specialized in spotting fine vertical phase anomalies, spectral gaps, or boundary artifacts characteristic of AI-cloned audio.

| Layer Stage | Layer Type | Specifications & Configurations | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Input** | Tensor Image | Normalized Mel-Spectrogram Matrix | `[B, 1, 128, 94]` |
| **Block 1** | 2D Convolution | 32 Filters, Kernel=3 * 3, Stride=1, Padding=1 | `[B, 32, 128, 94]` |
| | Batch Norm | 2D Batch Normalization (Stability & Speed) | `[B, 32, 128, 94]` |
| | Activation | ReLU (Rectified Linear Unit) | `[B, 32, 128, 94]` |
| | Max Pooling | Kernel Size=2 * 2, Stride=2 * 2 | `[B, 32, 64, 47]` |
| **Block 2** | 2D Convolution | 64 Filters, Kernel=3 * 3, Stride=1, Padding=1 | `[B, 64, 64, 47]` |
| | Batch Norm | 2D Batch Normalization | `[B, 64, 64, 47]` |
| | Activation | ReLU | `[B, 64, 64, 47]` |
| | Max Pooling | Kernel Size=2 * 2, Stride=2 * 2 | `[B, 64, 32, 23]` |
| **Block 3** | 2D Convolution | 128 Filters, Kernel=3 * 3, Stride=1, Padding=1 | `[B, 128, 32, 23]` |
| | Batch Norm | 2D Batch Normalization | `[B, 128, 32, 23]` |
| | Activation | ReLU | `[B, 128, 32, 23]` |
| | Max Pooling | Kernel Size=2 * 2, Stride=2 * 2 | `[B, 128, 16, 11]` |
| **Pooling** | Adaptive Avg | Forced Dimensional Spatial Collapse down to 4 * 4 | `[B, 128, 4, 4]` |
| **Flatten** | Tensor Reshape | Reshape matrix grid into a 1D flat vector | `[B, 2048]` |
| **Dense 1** | Fully Connected | Linear Input ($2048 \rightarrow 256$) + ReLU | `[B, 256]` |
| | Regularization | Dropout Layer (Rate = $0.40$ to prevent overfitting) | `[B, 256]` |
| **Output** | Linear Classifier | Fully Connected ($256 \rightarrow 1$ Unscaled Logit Value) | `[B, 1]` |

---

## Performance & Verification Report

Following validation and training convergence over the variable-length Fake-or-Real (FoR) split testing partition, the audited metrics comfortably surpassed all baseline targets:

### Core Performance Metrics
* **Overall Test Accuracy:** **89.94%** *(Project Target: $\ge$ 80%)*
* **Equal Error Rate (EER):** **10.07%** *(Project Target: $\le$ 12%)* — Calculated dynamically using Scipy's optimized **BrentQ root-finding method**, representing the strict operational intersection where FAR equals FRR.
* **F1-Score Metric:** **90.15%** *(Project Target: $\ge$ 80%)*
* **Genuine Per-Class Accuracy:** **89.93%** *(Project Target: $\ge$ 75%)*
* **Deepfake Per-Class Accuracy:** **89.96%** *(Project Target: $\ge$ 75%)*
* **Optimal Decision Boundary Logit:** **0.0002** ($\approx \mathbf{0.5000}$ Sigmoid Probability), validating the mathematical calibration of our classification threshold.

* *The extremely small variance (**0.03%**) between per-class metrics indicates that the network is exceptionally well-balanced and free from majority-class prediction bias.*

---

## Local & Web Deployment

### Prerequisites
Ensure your local environment includes the core operational dependencies. Install them via your console:
```bash
pip install streamlit torch librosa numpy scikit-learn
```

Local Standalone Inference Script
To test isolated voice samples directly from your terminal, run:
```bash
python predict.py path/to/your_voice_sample.wav
```
---


Running the Streamlit Web Application
```bash
streamlit run app.py
```
---

## Website URL
The url of the deployed website in which you can upload an audio file as input. 
Returns whether it is Genuine (Human) or
Deepfake (AI-Generated), along with the
confidence score.

```bash
https://deepfakeaudiodetection-uoufmgdtqpfp5hm8vcbpzp.streamlit.app/
```
