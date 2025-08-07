# Micro-Doppler Based Target Classification

## Problem Statement

To design and implement a radar-based system capable of classifying drones and birds using micro-Doppler signature analysis. The system aims to enhance situational awareness by reliably differentiating between drones and birds, leveraging the unique time-frequency features of their motion patterns for improved security and surveillance applications.

## Solution

Creating a ML model which uses the micro-doppler effects of the object for classification. And instead of rellying on the old traditional methods, using of CWT to get a better accuracy model.

## Our Idealogy

In traditional models, STFT are used for the conversion time domain into frequency domain which will cause some loss of information during the conversion. And they also analyze the entire spectrogram image to make the predicition. In our approach we overcome the loss of information loss during the conversion by Continous Wavelet transformation and also instead of focusing on the entire image. We extract important features from the image and make classification based on those features which reduces the computational expenses.

## Data Set Introduction

### 🛰️ Radar Setup
- **Radar Type**: FMCW radar (used for detecting range and motion)
- **Location**: Data was collected in real-time by KTH Royal Institute of Technology, Sweden

### ⚙️ Radar Configuration
- **Bandwidth**: 160 MHz (Gives high accuracy in range detection)
- **Pulse Rate Frequency**: 17 kHz (Radar sends 17,000 pulses per second)
- **Scan Time & Angle**: Scans 18° horizontally in 50 ms (Each scan captures objects within a narrow field of view)

### 🧭 Azimuth Sampling
- 256 azimuth samples per scan
- The 18° scan area is split into 256 small horizontal slices
- This gives fine directional resolution

### 🧮 Data Per Scan Segment
Each azimuth sample includes complex values that represent reflected signals (phase + magnitude).

Each scan segment has:
- 5 segments (like 5 snapshots over time or angle)
- Each segment = 256 azimuths

So: 5 segments × 256 azimuths = 1280 complex values

### 📐 Shape of One Radar Segment
When stored in an array: (1280, 25)
- 1280 rows = 5 segments × 256 azimuths
- 25 columns = 25 scan segments (time or angle slices)

## Data Cleaning and Processing from Raw Data Source

Each radar segment consists of 1280 complex samples, representing 5 range bins and 256 azimuth sweeps. These segments are processed to extract meaningful features for a machine learning model to distinguish between birds and drones.

### 1. Objective
The preprocessing script performs the following key operations:
- Loads and structures radar segment data
- Filters out irrelevant or underrepresented classes
- Labels data as either bird (0) or drone (1)
- Splits segment data into individual azimuth sweeps
- Saves processed data for model training and evaluation

### 2. Dataset Description
- **Source**: .npy file derived from .mat radar data
- **Original Format**: 130 x 6 matrix where each row is a measurement
- **Segment Details**:
   - Each radar segment: 1280 complex values
   - Represents: 5 range bins × 256 azimuth sweeps
   - Time per scan: 15 ms across 3.9° field of view
- **Column Descriptions**:
   - Target Type – Type of object (e.g., D1, seagull, human_run)
   - Segment Data – Complex-valued matrix (1280×N)
   - Range (meters) – Distance to target center
   - Time (seconds) – Timestamp of each segment
   - Dataset Split – Indicates training, validation, or test set
   - Edge Indicator – Whether target is near the field of view edge

### 3. Preprocessing Workflow
**Step 1: Data Loading**
- The .npy file containing all radar measurements is loaded
- It is converted to a structured DataFrame with named columns

**Step 2: Filtering and Labeling**
- Only relevant target types are retained (drones and birds)
- Human and corner reflector classes are excluded
- The target classes are mapped:
   - Birds: Labelled as 0
   - Drones: Labelled as 1

**Step 3: Saving Class-Wise Segment Data**
- Segments are saved into two directories:
   - Bird_segment_only_data
   - Drone_segment_only_data
- Each file contains a complete 1280×1 complex array for one segment

**Step 4: Splitting Segments into Azimuth Sweeps**
- Each segment (1280×1) is reshaped into a 5×256 matrix
- Each of the 256 azimuth sweeps is saved as a separate file (1×256)
- Files are stored in directories such as:
   - Bird_single_segment_data/bird1/bird1_1seg.npy
   - Drone_single_segment_data/drone56/drone56_3seg.npy

**Step 5: Validation**
- Random segment and sweep files are converted to CSV format
- Shapes and content are checked for correctness and consistency

**Step 6: Segment Counting**
- Total number of individual azimuth sweep files is computed
- This confirms the number of usable samples for training/testing

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

**Step 1: Load Radar Segment**
- Load .npy file containing radar signal segments
- Example file: drone7_segment_2.npy

**Step 2: Preprocess Segment**
- Extract and flatten the segment data
- Ensure data is in complex format (needed for wavelet transform)

### Why this Conversion
As the collected data's are Time Domain Data we transform them into frequency domain data:

- **STFT** (Short term fourier transform) → Spectogram, widely used, but fixed resolution and loss information during the conversion
- **ChirpLet Transformation** → Better for chirping signals, better accuracy, feature extraction but slower, complex and computational expensive
- **CWT** → Captures both transient and steady features, Adaptive time-frequency resolution, Less computationally expensive, More interpretable and easier to implement

So based on the research papers and faculty guidance we opted to choose between chirplet transformation and CWT (Continuous Wavelet Transform).
And We went for CWT because:
- Simpler and More Robust for Non-Chirping Signals
- Lower Computational Complexity

### How the conversion happens (we done conversion for a single segment only):

✅ **Formula for CWT**:
CWTx(a,b) = 1/|a| ∫-∞∞ x(t) ψ*((t-b)/a)dt

Where:
- x(t): Input signal
- ψ: Mother wavelet (e.g., Morlet)
- a: Scale (inversely related to frequency)
- b: Time shift
- ψ*: Complex conjugate of the wavelet

✅ **Conversion of (1, 1280) .npy segment to CWT image**:

1. Flatten the segment
    - From shape (1, 1280) → segment.flatten() → (1280,)

2. Apply CWT
    ```python
    coefficients, freqs = pywt.cwt(segment, scales=np.arange(1, 128), wavelet='cmor')
    ```
    - coefficients shape → (127, 1280)
    - (127 scales × 1280 time steps)

3. Take Magnitude
    ```python
    cwt_image = np.abs(coefficients)
    ```

4. Visualize
    ```python
    plt.imshow(cwt_image, cmap='viridis', aspect='auto')
    ```
    → This gives a Scalogram (like a spectrogram)

**Step 3: Apply CWT**
- Use Complex Morlet wavelet (cmor)
- Define a range of scales (1 to 127)
- Apply pywt.cwt() to get CWT coefficients
- Take the magnitude of the coefficients

**Step 4: Save CWT Image**
- Visualize CWT result using matplotlib
- Save the image as .png in a folder (e.g., bird/)

### Step 5: Feature Extraction from Images
Load CWT Spectrogram Images from two folders:
- Drone: contains drone spectrogram images
- Bird: contains bird spectrogram images

Process Each Image:
- Load as grayscale using OpenCV
- Normalize pixel values
- Compute power spectrum (sum of intensities along frequency axis)

Extract features:
- **Mean Peak Frequency**
   - Shows dominant motion frequencies
   - Birds have periodic flapping → multiple peaks
   - Drones have more constant rotor-induced frequencies → sharper peaks
- **Bandwidth**
   - Measures spread of frequency
   - Birds have broader bandwidth due to wing flapping
   - Drones show narrower bandwidth with stable rotors
- **Spectral Centroid**
   - Represents center of frequency mass
   - Drones usually have a higher centroid (high-frequency blades)
   - Birds have lower centroid due to natural wing motion
- **Spectral Entropy**
   - Indicates complexity/randomness
   - Bird motion is more random, hence higher entropy
   - Drone signals are structured, lower entropy
- **Total Energy**
   - Reflects overall signal strength
   - Drones often have stronger, more continuous energy
   - Birds show pulsating energy patterns

Other Features we may consider:
- Temporal features: How features change over time
- Higher-order statistics: Skewness, kurtosis of the power spectrum

Label the feature set as either 'Drone' or 'Bird'.

Save all features to a CSV file: `spectrogram_features.csv`

### Step 6: Data Preparation
- Load the CSV file
- Drop unnecessary column: image_path
- Encode labels:
   - 'Drone' → 1
   - 'Bird' → 0

### Step 7: Train-Test Split
- Features (X) = All columns except class
- Labels (y) = class column
- Split:
   - 80% for training
   - 20% for testing

### Step 9: Model Training
Model used: Random Forest Classifier

*We tried with various other models and all the drawbacks and it's accuracy were mentioned in the Project Documentation.pdf and after careful consideration we went with random forest classifier because it gives us the better accuracy and also prevents overfitting whereas other models mostly suffer with the overfitting problem*

Parameters:
- n_estimators = 100
- random_state = 42

Train on training set using fit().

### Step 10: Model Evaluation
Make predictions on the test set.

Metrics calculated:
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix (Visualized using seaborn heatmap)

## Conclusion

The model has been developed and the final project and deployment method that we used is mentioned in the project documentation with the architecture.

## 👥 Contributors
- [Dharshini](https://github.com/DharshiniRaji)
- [Mahima](https://github.com/Caeruleaphile08)
- [Prabha](https://github.com/prabhaM07)
