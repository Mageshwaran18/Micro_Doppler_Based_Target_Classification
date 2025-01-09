from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import joblib
from scipy.signal import find_peaks
from scipy.stats import entropy
import pywt
import time


# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
MODEL_FILENAME = "random_forest_model.pkl"
rf_classifier = joblib.load(MODEL_FILENAME)

# Folder to save temporary images and files
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Function to perform CWT on a segment
def perform_cwt(segment, scales, wavelet):
    coefficients, _ = pywt.cwt(segment, scales, wavelet)
    return np.abs(coefficients)

# Function to save CWT spectrogram as an image
def save_cwt_image(segment, scales, wavelet, save_path):
    cwt_result = perform_cwt(segment, scales, wavelet)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        cwt_result, extent=[0, cwt_result.shape[1], scales[-1], scales[0]],
        cmap='viridis', aspect='auto'
    )
    plt.colorbar(label='Magnitude')
    plt.title('CWT Spectrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Function to extract frequency features from spectrogram
def extract_frequency_features(image):
    normalized_image = image / 255.0
    power_spectrum = np.sum(normalized_image, axis=1)
    num_rows, _ = image.shape
    frequencies = np.linspace(0, 0.5, num_rows)

    features = {}
    peak_indices = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.1)[0]
    if len(peak_indices) > 0:
        peak_freqs = frequencies[peak_indices]
        features['mean_peak_frequency'] = np.mean(peak_freqs)
    else:
        features['mean_peak_frequency'] = 0

    power_sum = np.sum(power_spectrum)
    normalized_spectrum = power_spectrum / power_sum
    spectral_variance = np.sum(normalized_spectrum * (frequencies - np.mean(frequencies)) ** 2)
    features['bandwidth'] = np.sqrt(spectral_variance)

    spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    features['spectral_centroid'] = spectral_centroid

    features['spectral_entropy'] = entropy(power_spectrum + 1e-12)
    features['total_energy'] = np.sum(power_spectrum)

    return features

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    start_time = time.time()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Load the NumPy file
        data = np.load(file_path, allow_pickle=True)
        segment = np.array(data[0], dtype=complex)

        # Perform CWT and save the spectrogram image
        image_path = os.path.join(PROCESSED_FOLDER, 'spectrogram.png')
        scales = np.arange(1, 128)
        wavelet = 'cmor'
        save_cwt_image(segment, scales, wavelet, image_path)

        # Load the saved image for feature extraction
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        frequency_features = extract_frequency_features(image)

        # Prepare features for prediction
        features_df = pd.DataFrame([frequency_features])
        prediction = rf_classifier.predict(features_df)[0]

        # Calculate processing time
        processing_time = time.time() - start_time

        # Return the result
        result = "Bird" if prediction == 0 else "Drone"
        return jsonify({
            "prediction": result,
            "processing_time": f"{processing_time:.2f} seconds"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)