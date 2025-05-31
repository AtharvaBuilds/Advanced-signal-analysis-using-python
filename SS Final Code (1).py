import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import keyboard
import sys

# Path to the zip file
zip_filename = r'C:\Users\OMEN\Desktop\new ss dataset.zip'

# Extract files from zip and list them
with ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('extracted_wav_files')

# Get a list of .wav files
wav_files = [f for f in os.listdir('extracted_wav_files') if f.endswith('.wav')]
current_file_index = 0

def analyze_wav(file_path):
    """Analyzes a .wav file and plots PSD, ESD, autocorrelation, and convolution."""
    sample_rate, audio_data = wavfile.read(file_path)
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize
    
    # PSD
    frequencies, psd = signal.welch(audio_data, sample_rate)
    esd = psd * frequencies  # ESD is frequency-weighted PSD

    # Autocorrelation
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]

    # Convolution (Low-pass filter)
    sos = signal.butter(10, 1000, 'lp', fs=sample_rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_data)

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot PSD
    plt.subplot(3, 2, 1)
    plt.semilogy(frequencies, psd)
    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")

    # Plot ESD
    plt.subplot(3, 2, 2)
    plt.semilogy(frequencies, esd)
    plt.title("Energy Spectral Density (ESD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Energy/Frequency (dB/Hz)")

    # Plot Autocorrelation
    plt.subplot(3, 2, 3)
    plt.plot(autocorr)
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")

    # Plot Original vs Filtered Signal
    plt.subplot(3, 2, 4)
    plt.plot(audio_data, label="Original Signal", alpha=0.5)
    plt.plot(filtered_audio, label="Filtered Signal", alpha=0.75)
    plt.title("Original vs Filtered Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    global current_file_index
    while True:
        print(f"Analyzing file: {wav_files[current_file_index]}")
        analyze_wav(os.path.join('extracted_wav_files', wav_files[current_file_index]))

        # Go to the next audio file
        current_file_index = (current_file_index + 1) % len(wav_files)

if __name__ == "__main__":
    main()