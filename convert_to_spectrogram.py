import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
matplotlib.use('Agg')


''' LINEAR SPECTROGRAMS '''
def create_linear_spectrogram(filename):
    sample_rate, samples = wavfile.read(datasetdir+f'processed-audio/{filename}')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    
    plt.axis([0,max(times), 0,max(frequencies)])
    plt.axis('off')
    plt.specgram(samples, NFFT=256, Fs=sample_rate, noverlap=256*0.85)
    plt.savefig(datasetdir + f"processed-image/linear_{filename[:-4]}.png")
    plt.show()


''' MEL SPECTROGRAMS '''
def create_mel_spectrogram(filename):

    #1. 
    #  sig, fs = librosa.load(f'{datasetdir}AudioWAV/{filename}')   
    
    y, sample_rate=librosa.load(f'{datasetdir}processed-audio/{filename}')
    save_path = datasetdir + f'processed-image/{filename[:-4]}.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    #1.
    #  S = librosa.feature.melspectrogram(y=sig, sr=fs)
    # librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    # mel_spect = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=2048, hop_length=1024)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=256, hop_length=128, fmax=8000)

    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
    
    
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()


def mel_using_sklearn(filename):

    samples, sample_rate = librosa.load(f'{datasetdir}processed-audio/{filename}')   
    mfcc = librosa.feature.mfcc(samples, sr=sample_rate)
    
    # mfcc = preprocessing.scale(mfcc, axis=1)
    librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')

    save_path = datasetdir + f'processed-image/mfcc_{filename[:-4]}.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()



datasetdir="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset2/"
audio_clips = os.listdir(datasetdir+'processed-audio/')


# create_linear_spectrogram(audio_clips[0])
# create_mel_spectrogram(audio_clips[3])
# mel_using_sklearn(audio_clips[0])

k=1
for filename in audio_clips:
    print("Processing file no.", k)
    create_mel_spectrogram(filename)
    k+=1
