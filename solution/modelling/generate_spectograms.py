import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import os
import glob

def convert_spectograms(path_file):
    data, sampling_rate = librosa.load(path_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.min)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sampling_rate, )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{path_file.replace('.wav', '.png').replace('all', 'spectograms')}")
    plt.close()

if __name__ == "__main__":
    all_files = glob.glob("data/audio_data/all/*.wav")
    for file in all_files:
        convert_spectograms(file)