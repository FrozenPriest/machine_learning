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
        dictionary = {'Name': file, 'Type': folder}
        # short term fourier transform
        stft = numpy.abs(librosa.stft(sound_sample))
        # mfcc
        mfccs_mean = numpy.mean(librosa.feature.mfcc(y=sound_sample, sr=sound_rate, n_mfcc=40), axis=1)
        mfccs_std = numpy.std(librosa.feature.mfcc(y=sound_sample, sr=sound_rate, n_mfcc=40), axis=1)

        for i in range(0, 12):
            dictionary['mfccs_mean_' + str(i)] = mfccs_mean[i]
        for i in range(0, 12):
            dictionary['mfccs_std_' + str(i)] = mfccs_std[i]
        # chroma
        chroma_mean = numpy.mean(librosa.feature.chroma_stft(S=stft, sr=sound_rate), axis=1)
        chroma_std = numpy.std(librosa.feature.chroma_stft(S=stft, sr=sound_rate), axis=1)

        for i in range(0, 12):
            dictionary['chroma_mean_' + str(i)] = chroma_mean[i]
        for i in range(0, 12):
            dictionary['chroma_std_' + str(i)] = chroma_std[i]
        # melspectrogram
        mel_mean = numpy.mean(librosa.feature.melspectrogram(sound_sample, sr=sound_rate), axis=1)
        mel_std = numpy.std(librosa.feature.melspectrogram(sound_sample, sr=sound_rate), axis=1)

        for i in range(0, 12):
            dictionary['mel_mean_' + str(i)] = mel_mean[i]
        for i in range(0, 12):
            dictionary['mel_std_' + str(i)] = mel_std[i]
        # spectral contrast
        contrast_mean = numpy.mean(librosa.feature.spectral_contrast(S=stft, sr=sound_rate), axis=1)
        contrast_std = numpy.std(librosa.feature.spectral_contrast(S=stft, sr=sound_rate), axis=1)

        for i in range(0, 6):
            dictionary['contrast_mean_' + str(i)] = contrast_mean[i]
        for i in range(0, 6):
            dictionary['contrast_std_' + str(i)] = contrast_std[i]

        tonnetz_mean = numpy.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound_sample), sr=sound_rate), axis=1)
        tonnetz_std = numpy.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound_sample), sr=sound_rate), axis=1)
        for i in range(0, 6):
            dictionary['tonnetz_mean_' + str(i)] = tonnetz_mean[i]
        for i in range(0, 6):
            dictionary['tonnetz_std_' + str(i)] = tonnetz_std[i]

        music_data_dictionary.append(dictionary)
    print(f"Folder checked!!")
print(music_data_dictionary)


df = pandas.DataFrame(music_data_dictionary)

# df.set_index('Name')
df.to_csv(result_path, sep=';', index=False)
