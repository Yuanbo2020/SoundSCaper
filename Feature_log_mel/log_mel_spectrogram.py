import pickle
import sys, os, h5py, time, librosa, torch
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def save_pickle(file, dict):
    with open(file, 'wb') as f:
        pickle.dump(dict, f)


def run_jobs():
    source_path = os.path.join(os.getcwd(), 'Dataset')

    source_wav_path = os.path.join(os.getcwd(), 'Dataset_wav')

    output_feature_dir = source_path + "_mel"
    create_folder(output_feature_dir)

    mel_bins = 64
    sample_rate = 16000
    fmax = int(sample_rate / 2)
    fmin = 50
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    window_size = 512
    hop_size = 160

    spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                        win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                                        freeze_parameters=True)

    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                        freeze_parameters=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    spectrogram_extractor.to(device)
    logmel_extractor.to(device)

    for (n, audioname) in enumerate(os.listdir(source_wav_path)):
        audio_path = os.path.join(source_wav_path, audioname)

        feature_filename = audioname.replace('.wav', '.npy')
        output_feature = os.path.join(output_feature_dir, feature_filename)

        audiodata, fs = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        # print(audiodata.shape, fs)  # (480000,) 16000

        spectrogram = spectrogram_extractor(move_data_to_device(audiodata[None, :], device))

        logmel = logmel_extractor(spectrogram)  # torch.Size([1, 1, 3001, 64])
        # print(logmel.shape)

        logmel = logmel[0, 0].data.cpu().numpy()
        print(n, feature_filename, logmel.shape)  # torch.Size([3001, 64])
        # 0 fold_0_participant_10001_stimulus_02.npy (3001, 64)

        np.save(output_feature, logmel)


def main(argv):
    run_jobs()


if __name__=="__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















