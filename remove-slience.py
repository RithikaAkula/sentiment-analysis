import os
import librosa
import soundfile as sf


def remove_silence(filename):

    y, sr=librosa.load(f'{datasetdir}AudioWAV/{filename}')
    op=librosa.effects.trim(y, top_db=25)

    save_path = datasetdir + f'processed-audio/{filename[:-4]}.wav'

    sf.write(save_path, op[0], sr)



datasetdir="D:/4-2/MAJOR_PROJECT/sentiment-analysis/dataset2/"
audio_clips = os.listdir(datasetdir+'AudioWAV/')

# remove_silence(audio_clips[2])

k=1
for filename in audio_clips:
    print("Processing file no.", k)
    try:
        remove_silence(filename)
    except:
        print(filename)
        continue
    k+=1