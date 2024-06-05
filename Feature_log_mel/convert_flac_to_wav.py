import os, sys, librosa
import soundfile


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def write_audio_wav(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


sample_rate = 44100
source_flac_path = os.path.join(os.getcwd(), 'Dataset')
output_dir = source_flac_path + '_wav'
create_folder(output_dir)

for audio_name in os.listdir(source_flac_path):
    wav_file = os.path.join(output_dir, audio_name.replace('.flac', '.wav'))

    flac_file = os.path.join(source_flac_path, audio_name)
    audiodata, fs = librosa.core.load(flac_file, sr=sample_rate, mono=True)
    # print(audiodata.shape, fs, len(audiodata) / fs)  # (1323000,) 44100 30.0

    write_audio_wav(wav_file, audiodata, sample_rate)










