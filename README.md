# Radar-Based Drone and Bird Classification System

## Problem Statement

To design and implement a radar-based system capable of classifying drones and birds using micro-Doppler signature analysis. The system aims to enhance situational awareness by reliably differentiating between drones and birds, leveraging the unique time-frequency features of their motion patterns for improved security and surveillance applications.

## Solution

Creating a ML model which uses the micro-doppler effects of the object for classification. And instead of rellying on the old traditional methods, using of CWT to get a better accuracy model.

## Our Idealogy

In traditional models, STFT are used for the conversion time domain into frequency domain which will cause some loss of information during the conversion. And they also analyze the entire spectrogram image to make the predicition. In our approach we overcome the loss of information loss during the conversion by Continous Wavelet transformation and also instead of focusing on the entire image. We extract important features from the image and make classification based on those features which reduces the computational expenses.

## Data Set Introduction

Data Set was a realtime data collected by KTH Royal Institute of Technology, Sweden.

### Radar Specifications (FMCW)
- Bandwidth: 160 MHZ (range & frequency)
- Pulse rate frequency: 17 KHz
- Scan angle: 18° in 50 ms
- 256 azimuth samples at different ranges
- Each sample: 5 segments with 256 azimuths measured with complex values
- [ Scan segments are thus chunk of data representing the slice of the radar is observation over time (or) angle] [Azimuth samples refers to the (horizontal angle) direction]
   ```
   seg 1 [CV₁,CV₂,CV₃...CV₂₅₆]
   seg 5 [CV₁,CV₂,CV₃...CV₂₅₆]
   ```
- Total samples: 1280
- Array dimension: (1280,25)

## Data Cleaning and Processing

### 1. Objective
The preprocessing script performs:
- Loads and structures radar segment data
- Filters out irrelevant classes
- Labels data as bird (0) or drone (1)
- Splits segment data into azimuth sweeps
- Saves processed data for model training

### 2. Dataset Description
- Source: .npy file from .mat radar data
- Format: 130 x 6 matrix per measurement
- Segment details: 1280 complex values
- Time per scan: 15 ms across 3.9° field of view

### 3. Preprocessing Workflow
1. Data Loading
2. Filtering and Labeling
3. Saving Class-Wise Segment Data
4. Splitting Segments into Azimuth Sweeps
5. Validation
6. Segment Counting

### 4. Output Directory Structure
```
Micro_Doppler_Based_Target_Classification/
│
├── Drone_segment_only_data/
│   ├── drone1.npy
│   └── ...
│
├── Bird_segment_only_data/
│   ├── bird1.npy
│   └── ...
│
├── Drone_single_segment_data/
│   ├── drone1/
│   │   ├── drone1_1seg.npy
│   │   └── ...
│
├── Bird_single_segment_data/
│   ├── bird1/
│   │   ├── bird1_1seg.npy
│   │   └── ...
```

## Model Building and Training Process

### Input Data Generation
1. Load Radar Segment
2. Preprocess Segment
3. Apply CWT
4. Save CWT Image
5. Feature Extraction
6. Data Preparation
7. Train-Test Split
8. Model Training
9. Model Evaluation

### Why CWT Over Other Transformations
- Simpler and More Robust for Non-Chirping Signals
- Lower Computational Complexity
- Better feature capture capabilities

#### CWT Formula
```
CWTx(a,b) = 1/|a| ∫-∞∞ x(t) ψ*((t-b)/a)dt
```

## Model Selection
Random Forest Classifier was chosen after careful consideration of various models due to:
- Better accuracy
- Prevention of overfitting
- Robust performance

## For Brief Documentation Refer 
- Brief End to End Documentation

## Contributors
- [Dharshini](https://github.com/DharshiniRaji)
- [Mahima](https://github.com/Caeruleaphile08)
- [Prabha](https://github.com/prabhaM07)
- [Rijas](https://github.com/abdulrijas)
- [Bala Mugesh](https://github.com/BMugesh)