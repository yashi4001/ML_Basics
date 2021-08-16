import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np

file="your-summer-day-5448.wav"

#waveform
signal,sr=librosa.load(file,sr=22050) #signal will be a numpy array which will have no.of values=sr*duration of sound track
librosa.display.waveplot(signal,sr=sr) #visualizing the wave
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#time domain->frequency domain(fourier tranform)
fft=np.fft.fft(signal) #np array

magnitude= np.abs(fft) #indicates contrib of each frequency to the sound
frequency=np.linspace(0,sr,len(magnitude))
left_frequency=frequency[:int(len(frequency)/2)]
left_magnitude=magnitude[:int(len(magnitude)/2)]
plt.plot(left_frequency,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#get spectogram(amplitude as function of freq and time)
n_fft=2048 #no.of sample in each fft
hop_length=512 #amount of shift to next fft to the right
stft=librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
spectrogram=np.abs(stft)
log_spectrogram=librosa.amplitude_to_db(spectrogram) #converting amplitude to decibel

librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length) #specshow helps to visualize spectogram like data(x axis, y axis and color label)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar() #amplitude will be displayed by color
plt.show()

#mfccs
MFCCS=librosa.feature.mfcc(signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=13)
librosa.display.specshow(MFCCS,sr=sr,hop_length=hop_length) #specshow helps to visualize spectogram like data(x axis, y axis and color label)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar() #amplitude will be displayed by color
plt.show()





