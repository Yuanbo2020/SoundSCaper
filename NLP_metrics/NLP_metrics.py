import os

source_path = os.path.join(os.getcwd(), 'Source')

GT_labels = []
filename = 'Expert_description.txt'
filepath = os.path.join(source_path, 'From QQ', filename)
with open(filepath, 'r') as f:
    for num, line in enumerate(f.readlines()):
        # print([line])
        each = line.split('\n')[0].replace('\"', '')
        # print(num, each)
        GT_labels.append(each)


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def save_results(filename, score_list):
    score_names = ['BLEU', 'ROUGE-L Recall', 'ROUGE-L Precision', 'ROUGE-L F1 Score', 'METEOR', 'CIDEr']

    dir_name = 'NLP_results'
    dirpath = os.path.join(os.getcwd(), dir_name)
    create_folder(dirpath)

    output_filename = os.path.join(dirpath, 'NLP_results_' + filename)
    with open(output_filename, 'w') as f:
        f.write('\t'.join(score_names) + '\n')
        for each in score_list:
            f.write('\t'.join([str(sub_each) for sub_each in each]) + '\n')


def cal_scores(filename, Soundscaper_sentences, GT_labels):
    from nltk.translate.bleu_score import sentence_bleu
    from pycocoevalcap.bleu.bleu import Bleu
    from rouge import Rouge
    import nltk
    nltk.download('wordnet')
    from nltk.translate import meteor_score
    from pycocoevalcap.cider.cider import Cider
    scorer = Cider()

    score_list = []
    for num, (reference, candidate) in enumerate(zip(GT_labels, Soundscaper_sentences)):
        current = []
        print(num, reference)
        print(num, candidate)
        print([reference.split()])
        print(candidate.split())
        score_BLEU = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))  # BLEU-4
        print(score_BLEU)
        # 0.19792270631314585
        current.append(score_BLEU)

        rouger = Rouge()
        score_rouger = rouger.get_scores(candidate, reference)
        print(score_rouger)
        current.append(score_rouger[0]['rouge-l']['r'])
        current.append(score_rouger[0]['rouge-l']['p'])
        current.append(score_rouger[0]['rouge-l']['f'])

        score_meteor = meteor_score.meteor_score([reference.split()], candidate.split())
        print(score_meteor)
        current.append(score_meteor)

        gts = {"184321": [reference], "null": ["0"]}
        res = {"184321": [candidate], "null": ["0"]}
        (score, scores) = scorer.compute_score(gts, res)
        # print('cider = %s' % score)
        score_cider = scores[0]
        print(score_cider)
        current.append(score_cider)

        score_list.append(current)
        print('\n')

        save_results(filename, score_list)


Soundscaper_sentences = []


##########################################################################################

filename = 'result_test_sr.json'
filepath = os.path.join(source_path, filename)

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
    values_sorted.append(values[ids.index(each)])
Soundscaper_sentences = values_sorted

print(filename)
cal_scores(filename.replace('.json', '.txt'), Soundscaper_sentences, GT_labels)



filename = 'LocalAFT.txt'
filepath = os.path.join(source_path, filename)
num=0
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
            print(each)
            Soundscaper_sentences.append(each)
# print(Soundscaper_sentences)

print(filename)
cal_scores(filename, Soundscaper_sentences, GT_labels)



filename = 'Soundscaper.txt'
filepath = os.path.join(source_path, 'From QQ', filename)
num=0
with open(filepath, 'r', encoding="utf8", errors='ignore') as f:
    for _, line in enumerate(f.readlines()):
        print([line])
        each = line.split('\n')[0].replace('\"', '')
        if len(each):
            Soundscaper_sentences.append(each)

print(filename)
cal_scores(filename, Soundscaper_sentences, GT_labels)



# 1: GAMA
# 2: GAMA-IT
filename = 'GAMA-audio-caption.txt'
sub_dir = 'Gama'
filepath = os.path.join(source_path, sub_dir, filename)
GAMA_list = []
GAMA_list_IT = []
with open(filepath, 'r') as f:
    for num, line in enumerate(f.readlines()):
        if '1A: ' in line:
            each = line.split('\n')[0].split('1A: ')[-1]
            GAMA_list.append(each)
        if '2A: ' in line:
            each = line.split('\n')[0].split('2A: ')[-1]
            GAMA_list_IT.append(each)

assert len(GAMA_list)==len(GAMA_list_IT)


filename = 'Gama-IT-audio-caption.txt'
cal_scores(filename, GAMA_list_IT, GT_labels)


###################################################################################################################
# 1: GAMA
# 2: GAMA-IT
filename = 'GAMA-caption-AQ.txt'
sub_dir = 'Gama'
filepath = os.path.join(source_path, sub_dir, filename)
GAMA_list = []
GAMA_list_IT = []
with open(filepath, 'r') as f:
    for num, line in enumerate(f.readlines()):
        if '1A: ' in line:
            each = line.split('\n')[0].split('1A: ')[-1]
            GAMA_list.append(each)
        if '2A: ' in line:
            each = line.split('\n')[0].split('2A: ')[-1]
            GAMA_list_IT.append(each)

assert len(GAMA_list)==len(GAMA_list_IT)

filename = 'Gama-AQ-IT-audio-caption.txt'
cal_scores(filename, GAMA_list_IT, GT_labels)



###################################################################################################################
Qwen_sentences = []
filename = 'Qwen2-audio-caption.txt'
filepath = os.path.join(source_path, 'Qwen', filename)
with open(filepath, 'r') as f:
    for num, line in enumerate(f.readlines()):
        # print([line])
        each = line.split('\n')[0].split(':')[-1]
        if len(each):
            Qwen_sentences.append(each)
# print(Qwen_sentences)
cal_scores(filename, Qwen_sentences, GT_labels)




from rouge import Rouge
hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
rouger = Rouge()
scores = rouger.get_scores(hypothesis, reference)
# print(scores)


import nltk
nltk.download('wordnet')
from nltk.translate import meteor_score
generated_text = "This is some generated text.".split()
reference_texts = ["This is some generated text.".split()]
meteor = meteor_score.meteor_score(reference_texts, generated_text)
print("The METEOR score is:", meteor)


# CIDEr(Consensus-based Image Description Evaluation)
from pycocoevalcap.cider.cider import Cider
scorer = Cider()
gts={"184321": ["This is some generated text."], "184320": ["This"]}
res={"184321": ["This is some generated text."], "184320": ["This"]}
(score, scores) = scorer.compute_score(gts, res)
print('cider = %s' % score)



