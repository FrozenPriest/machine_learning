import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import datetime
import numpy
import pandas
import sklearn
from pandas import DataFrame

data_path = 'C:\\Users\\boral\\PycharmProjects\\machine_learning\\data'
result_path = 'C:\\Users\\boral\\PycharmProjects\\machine_learning\\dataset.csv'

print(os.listdir(data_path))

music_data_dictionary = []
#
# chroma
# mfcc
# mel
#

for folder in os.listdir(data_path):
    folder_path = data_path + '\\' + folder
    if not os.path.isdir(folder_path):
        continue
    print(f"Checking folder {folder}")
    for file in os.listdir(folder_path):
        print(file)
        file_path = folder_path + '\\' + file
        sound_sample, sound_rate = librosa.load(file_path, res_type='kaiser_fast')

        # short term fourier transform
        stft = numpy.abs(librosa.stft(sound_sample))
        # mfcc
        mfccs = numpy.mean(librosa.feature.mfcc(y=sound_sample, sr=sound_rate, n_mfcc=40).T, axis=0)
        # chroma
        chroma = numpy.mean(librosa.feature.chroma_stft(S=stft, sr=sound_rate).T, axis=0)
        # melspectrogram
        mel = numpy.mean(librosa.feature.melspectrogram(sound_sample, sr=sound_rate).T, axis=0)
        # spectral contrast
        contrast = numpy.mean(librosa.feature.spectral_contrast(S=stft, sr=sound_rate).T, axis=0)
        tonnetz = numpy.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound_sample), sr=sound_rate).T, axis=0)

        music_data_dictionary.append({
            'Name': file,
            'Type': folder,
            'chroma': chroma,
            'mfccs': mfccs,
            'mel': mel,
            'contrast': contrast,
            'tonnetz': tonnetz
        })
    print(f"Folder checked!!")
print(music_data_dictionary)


df = pandas.DataFrame(music_data_dictionary)

# df.set_index('Name')
df.to_csv(result_path, sep=';', index=False)
