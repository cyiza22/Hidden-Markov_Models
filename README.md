# Human Activity Recognition Using Hidden Markov Models

## Project Overview

This project implements a smartphone-based activity recognition system using Hidden Markov Models (HMM) to detect daily activities such as standing, walking, jumping, and sitting still. The system achieves **90.9% accuracy** and is designed for potential applications in elderly care monitoring, where continuous activity tracking can provide early warnings of health issues without compromising privacy.

**Authors:** Henriette Cyiza, Jeremiah Agbaje  
**Date:** October 30, 2025  
**Repository:** [https://github.com/cyiza22/Hidden-Markov_Models](https://github.com/cyiza22/Hidden-Markov_Models)

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Team Collaboration](#team-collaboration)

---

## Motivation

In elderly care facilities, continuous monitoring of daily activities is crucial for detecting falls and unusual behavior patterns. Unlike video surveillance, smartphone-based activity recognition preserves privacy while providing valuable health insights [2]. The core challenge is that while smartphones continuously measure motion through accelerometers and gyroscopes, the true activity remains hidden behind noisy sensor measurements. Hidden Markov Models naturally address this by treating activities as hidden states and sensor readings as observations, incorporating temporal dependencies in the recognition process [1], [3].

---

## Dataset

### Data Collection Protocol

We collected **55 recordings** over two days using the Sensor Logger application on Tecno Pop 9 smartphones at 100 Hz sampling rate.

| Team Member | Phone | Sampling Rate | Activities | Files |
|-------------|-------|---------------|------------|-------|
| Henriette CYIZA | Tecno Pop 9 | 100 Hz | Standing, Walking | 28 |
| Jeremiah Agbaje | Tecno Pop 9 | 100 Hz | Jumping, Still | 27 |

**Dataset Statistics:**
- 55 recordings (5-10 seconds each)
- ~36,000 total data points
- Activities: Standing (12), Walking (16), Jumping (14), Still (13)

**Phone Positioning:**
- **Standing:** Phone at chest level
- **Walking:** Phone in right front pocket
- **Jumping:** Phone held at waist
- **Still:** Phone flat on the table

### Preprocessing Pipeline

1. **Extraction:** Automatically extracted accelerometer and gyroscope CSV files from ZIP archives
2. **Synchronization:** Merged sensor streams using `merge_asof` with 0.05s tolerance based on timestamps
3. **Validation:** Verified required columns, minimum duration (5s), and no timestamp gaps
4. **Result:** All 55 recordings passed validation

The synchronized data structure preserved: `time`, `accel_x/y/z`, `gyro_x/y/z`, `activity`, and `recording_id` for proper train/test splitting [4].

### Data Characteristics

Raw sensor visualizations confirmed distinct activity signatures:
- **Standing:** Minimal variation with small gyroscope fluctuations from body sway
- **Walking:** Clear ~2 Hz periodic patterns in both sensors
- **Jumping:** Extreme acceleration spikes (±5-10 m/s²)
- **Still:** Only sensor noise floor levels (±0.01 m/s²)

---

## Features

### Feature Engineering Strategy

We extracted **39 features** combining time-domain statistics and frequency-domain characteristics [4], [5].

#### Time-Domain Features (30 features)

**Per-axis statistics (24 features: 6 axes × 4 metrics):**
- **Mean:** Average sensor value; indicates gravity orientation for accelerometers
- **Standard Deviation:** Signal variability (still: ~0.01 m/s², walking: ~0.3 m/s², jumping: >1 m/s²)
- **Max/Min:** Capture extreme values; jumping produces peaks 5-10× larger than other activities
- **Range:** Max-min difference; highly discriminative for intensity separation

**Multi-axis aggregates (6 features):**
- **accel_sma:** Signal Magnitude Area - direction-independent movement intensity
- **gyro_sma:** Gyroscope SMA - our #1 most important feature (importance: 7.92)
- **accel_magnitude:** √(x²+y²+z²) - orientation-independent total acceleration

#### Frequency-Domain Features (9 features)

Applied FFT to accelerometer axes to extract:
- **Dominant frequency:** Peak frequency in spectrum (walking: ~2 Hz, jumping: ~1-1.5 Hz)
- **Spectral energy:** Σ|FFT(x)|² - quantifies total frequency content

**Rationale:** Time-domain features alone cannot distinguish random noise from structured periodicity. Walking and standing might have similar variance, but walking shows a sharp frequency peak at the step rate while standing shows a flat noise spectrum [4].

### Feature Importance

**Top 10 discriminative features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | gyro_sma | 7.92 |
| 2 | accel_z_min | 7.64 |
| 3 | gyro_x_std | 7.52 |
| 4 | accel_z_range | 7.42 |
| 5 | accel_magnitude | 7.26 |

**Key insight:** Gyroscope features dominated (5 of top 10), proving rotational motion is more discriminative than linear acceleration for these activities [5].

### Normalization

Applied Z-score standardization: z = (x-μ)/σ using training data statistics to:
- Equalize scales (accelerometers: ~0.01-10 m/s², gyroscopes: ~0.0001-0.001 rad/s)
- Center features at zero with unit variance
- Prevent numerical overflow in probability calculations
- Improve classifier performance

---

## Model Architecture

### HMM Structure

- **Hidden States (Z):** Four discrete activities (Standing, Walking, Jumping, Still)
- **Observations (X):** 39-dimensional normalized feature vectors
- **Parameters:**
  - Initial probabilities (π): [0.25, 0.25, 0.25, 0.25] - uniform start
  - Transition probabilities (A): Diagonal = 0.70 (self-transition), off-diagonal = 0.10 (switching)
  - Emission probabilities (B): Modeled as Gaussian: P(x|state) = N(x; μ, σ²)

### Training Approach

We used Gaussian Naive Bayes-inspired training due to limited data (44 training samples ÷ 4 classes ≈ 11 per class) [1]. This approach:
1. Computes class-wise mean μ and standard deviation σ for each feature
2. Calculates prior probabilities from class frequencies
3. Predicts via maximum log-likelihood: argmax[−0.5Σ((x−μ)/σ)² + log(π)]

### Viterbi Algorithm Concept

The system embodies Viterbi principles for finding the most likely state sequences [1]:
1. **Initialization:** δ₁(i) = log(πᵢ) + log(B(x₁|sᵢ))
2. **Recursion:** δₜ(j) = maxᵢ[δₜ₋₁(i) + log(Aᵢⱼ)] + log(B(xₜ|sⱼ))
3. **Backtrack** to recover the optimal path

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyiza22/Hidden-Markov_Models.git
   cd Hidden-Markov_Models
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
hmmlearn>=0.3.0
```

---

## Usage

### Running the Project

The entire project workflow is contained in a Jupyter notebook:

```bash
# Launch Jupyter Notebook
jupyter notebook

# Navigate to and open: notebooks/hmm_activity_recognition.ipynb
```

Alternatively, run directly from command line:

```bash
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

The notebook contains all steps:
- Data preprocessing and extraction from ZIP archives
- Feature extraction (time-domain and frequency-domain)
- Model training with HMM
- Model evaluation and visualization generation

---

## Results

### Overall Performance

- **Overall Accuracy:** 90.9%
- **Train/Test Split:** 80/20 (44 training, 11 testing samples)

### Per-Activity Results

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|----------|---------|-------------|-------------|----------|
| Standing | 2 | 1.000 | 1.000 | 1.000 |
| Walking | 4 | 0.750 | 1.000 | 0.909 |
| Jumping | 2 | 1.000 | 0.889 | 0.909 |
| Still | 3 | 1.000 | 1.000 | 1.000 |

### Analysis

- **Standing/Still/Jumping:** Perfect classification demonstrates clear feature separation
- **Walking:** One sample misclassified as jumping (75% sensitivity) - likely captured particularly vigorous stride resembling jumping's intensity
- **Confusion Matrix:** Strong diagonal with a single off-diagonal element (walking→jumping), no confusion between standing/still
- **Key Achievement:** Perfect classification of jumping (fall indicator) and still (prolonged immobility) - critical for elderly care applications

---

## Project Structure

```
Hidden-Markov_Models/
│
├── notebooks/
│   └── hmm_activity_recognition.ipynb      
│
├── results/
│   ├── evaluation_results.csv
│   └── feature_importance.csv
│
├── samples/                # 55  data samples collected with smartphones      
│
├── visualizations/         # visualizations from the outputs and results
│   ├── confusion_matrix.png
│   ├── emission_probabilities.png
│   ├── raw_sensor_data_visualization.png
│   └── transition_matrix.png
│
├── .gitignore
├── model_parameters.json
├── README.md
└── requirements.txt
```

---

## Team Collaboration

### Henriette Cyiza
- Data Collection: Standing (12), Walking (16)
- Feature Extraction
- Model Training
- Visualization: Raw sensor plots, confusion matrix, feature importance, transition matrix
- GitHub: Repository setup, code documentation, data organization
- Report: Sections 1-3, collaborative contribution to Sections 4-6

### Jeremiah Agbaje
- Data Collection: Jumping (14), Still (13)
- Model Evaluation
- Visualization: Supporting plots and analysis
- GitHub: README creation, file structure, 
- Report: Sections 1-3, collaborative contribution to Sections 4-6
---


