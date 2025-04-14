import os
import requests
from tqdm import tqdm
import torch
import numpy as np
import laion_clap
from clap_module.factory import load_state_dict
import librosa
import pyloudnorm as pyln


# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)



def clap_score(id2text, audio_path, audio_files_extension, sample_rate):
    batch_size = 64
    text_emb = {}
    for i in tqdm(range(0, len(id2text), batch_size)):
        batch_ids = list(id2text.keys())[i:i + batch_size]
        batch_texts = [id2text[id] for id in batch_ids]
        with torch.no_grad():
            embeddings = model.get_text_embedding(batch_texts, use_tensor=True)
        for id, emb in zip(batch_ids, embeddings):
            text_emb[id] = emb

    print('[EVALUATING GENERATIONS] ', audio_path)
    score = 0
    count = 0
    for id in tqdm(id2text.keys()):
        file_path = os.path.join(audio_path, str(id) + audio_files_extension)
        # print('file_path:', file_path)
        with torch.no_grad():
            audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1)  # unsqueeze (1,T)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            audio_embeddings = model.get_audio_embedding_from_data(x=audio, use_tensor=True)
        cosine_sim = \
            torch.nn.functional.cosine_similarity(audio_embeddings, text_emb[id].unsqueeze(0), dim=1, eps=1e-8)[0]
        score += cosine_sim
        count += 1

    return score / count if count > 0 else 0



def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


if __name__ == "__main__":
    # model_name = '630k-audioset-fusion-best.pt'

    model_name = 'music_speech_audioset_epoch_15_esc_89.98.pt'

    url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
    clap_path = os.path.join(os.getcwd(), 'load', 'clap_score', model_name)
    # model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
    # model = laion_clap.CLAP_Module(enable_fusion=True, device='mps')
    model = laion_clap.CLAP_Module(enable_fusion=True, device='cpu')

    # download clap_model if not already downloaded
    clap_model = model_name
    if not os.path.exists(clap_path):
        print('Downloading ', clap_model, '...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(clap_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))

    # fixing CLAP-LION issue, see: https://github.com/LAION-AI/CLAP/issues/118
    pkg = load_state_dict(clap_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()
    ##################################################################################################################

    source_path = 'Source'
    generated_path = os.path.join(os.getcwd(), 'audio_files')
    sample_rate = 44100

    dir_name = 'CLAP_' + model_name
    dirpath = os.path.join(os.getcwd(), dir_name)
    create_folder(dirpath)

    all_dict = {}

    # P-LocalAFT
    id2text = {}
    filename = 'P-LocalAFT.txt'
    filepath = os.path.join(source_path, filename)
    num = 0
    with open(filepath, 'r', encoding="utf8", errors='ignore') as f:
        for _, line in enumerate(f.readlines()):
            # print(_, line)
            if '.flac: ' in line:
                each = line.split('\n')[0].split('.flac: ')[-1]
            elif '.flac:' in line:
                each = line.split('\n')[0].split('.flac:')[-1]
            elif '.wav: ' in line:
                each = line.split('\n')[0].split('.wav: ')[-1]
            elif '.wav:' in line:
                each = line.split('\n')[0].split('.wav:')[-1]
            else:
                each = ''
            if len(each):
                # print(each, num)
                id2text[str(num + 1)] = each
                num += 1

    all_dict[filename] = id2text
    # print(len(id2text.keys()))
    # print(id2text.keys())

    # ConvNeXt-Trans
    id2text = {}
    filename = 'ConvNeXt-Trans.json'
    filepath = os.path.join(source_path, filename)
    num = 0

    import json
    with open(filepath) as f:
        d = json.load(f)

    ids = []
    values = []
    for key, value in d.items():
        # print(key.split('.')[0], value)
        ids.append(float(key.split('.')[0]))
        values.append(value)

    id_sorted = sorted(ids)
    values_sorted = []
    for each in id_sorted:
        # values_sorted.append(values[ids.index(each)])
        id2text[str(num + 1)] = values[ids.index(each)]
        num += 1

    all_dict[filename] = id2text
    # print(len(id2text.keys()))
    # print(id2text.keys())



    # human expert
    id2text = {}
    filename = 'Expert_description.txt'
    filepath = os.path.join(source_path, 'From QQ', filename)
    with open(filepath, 'r') as f:
        for num, line in enumerate(f.readlines()):
            # print([line])
            each = line.split('\n')[0].replace('\"', '')
            # print(num, each)
            id2text[str(num+1)] = each

    all_dict[filename] = id2text

    # Soundscaper
    id2text = {}
    filename = 'SoundScaper.txt'
    filepath = os.path.join(source_path, 'SoundScaper', filename)
    num=0
    with open(filepath, 'r', encoding="utf8", errors='ignore') as f:
        for _, line in enumerate(f.readlines()):
            print([line])
            each = line.split('\n')[0].replace('\"', '')
            if len(each):
                id2text[str(num + 1)] = each
                num += 1
    all_dict[filename] = id2text

    # GAMA
    id2text = {}
    filename = 'GAMA-audio-caption.txt'
    sub_dir = 'Gama'
    filepath = os.path.join(source_path, sub_dir, filename)
    num = 0
    with open(filepath, 'r') as f:
        for _, line in enumerate(f.readlines()):
            if '1A: ' in line:
                each = line.split('\n')[0].split('1A: ')[-1]
                id2text[str(num + 1)] = each
                num+=1
            if '2A: ' in line:
                # each = line.split('\n')[0].split('2A: ')[-1]
                # GAMA_list_IT.append(each)
                pass
    all_dict[filename] = id2text


    ###################################################################################################################
    Qwen_sentences = []
    filename = 'Qwen2-audio-caption.txt'
    filepath = os.path.join(source_path, 'Qwen', filename)
    id2text = {}
    num = 0
    with open(filepath, 'r') as f:
        for _, line in enumerate(f.readlines()):
            # print([line])
            each = line.split('\n')[0].split(':')[-1]
            if len(each):
                id2text[str(num + 1)] = each
                num += 1
    all_dict[filename] = id2text


    all_model_score = []
    names = []
    for filename, id2text in all_dict.items():
        names.append(filename)
        clap_scores = []
        for key, value in id2text.items():
            sub_key = {}
            sub_key[key] = value
            clp = clap_score(sub_key, generated_path, sample_rate=sample_rate, audio_files_extension='.wav')
            # print(key, ':', float(clp))
            clap_scores.append(float(clp))

        all_model_score.append(clap_scores)

    filename = os.path.join(dirpath, "CLAP" + '.txt')
    np.savetxt(filename, np.array(all_model_score).T)















