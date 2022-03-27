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
    sample_rate, samples = wavfile.read(projectdir+f'dataset/AudioWAV/{filename}.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.axis([0,max(times), 0,max(frequencies)])
    plt.axis('off')
    plt.specgram(samples, NFFT=256, Fs=sample_rate, noverlap=256*0.85)
    plt.savefig(projectdir + f"dataset/image/linear_{filename}.png")
    plt.show()


''' MEL SPECTROGRAMS '''
def create_mel_spectrogram(filename):
    sig, fs = librosa.load(f'{projectdir}dataset/AudioWAV/{filename}.wav')   
    save_path = projectdir + f'dataset/image/{filename}.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()



projectdir="D:/4-2/MAJOR_PROJECT/sentiment-analysis/"
audio_clips = os.listdir(projectdir+'dataset/AudioWAV/')


for filename in audio_clips:
    filename=filename[:-4]
    try:
        create_mel_spectrogram(filename)
    except:
        print(filename)
        continue
