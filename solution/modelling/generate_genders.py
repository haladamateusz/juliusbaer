import os
import shutil
import glob
from transformers import pipeline
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

os.makedirs('data/audio_data', exist_ok=True)

all_files = glob.glob('data/audio_data/*.wav')


classifier = pipeline('audio-classification', model='alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')
def get_predictions(classifier, files):
    preds = []
    print("i am working ")
    for x in files:
        # preds.append(classifier(inputs=[x]))
        print("*")
    return preds

preds = get_predictions(classifier=classifier, files=all_files)

with open('gender_classific.txt', 'w') as f:
    for file, pred in zip(all_files, preds):
        label = pred[0]['label']
        f.write(f"{label} {file}\n")
