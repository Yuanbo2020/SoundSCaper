import shutil
import os, pickle
import numpy as np

event_path = os.path.join(os.getcwd(), 'SoundAQnet_event_probability')
source_scene_path = os.path.join(os.getcwd(), 'SoundAQnet_scene_ISOPl_ISOEv_PAQ8DARs')

audio_names = []

for each in os.listdir(event_path):
    audio_names.append(each)
print(len(audio_names))


event_labels = ['Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds', 'Traffic',
                'Sounds of things', 'Vehicle', 'Bird', 'Outside	 rural or natural',
                'Environment and background', 'Speech', 'Music', 'Noise', 'Animal']

scene_labels = ['public_square', 'park', 'street_traffic',]

emotion = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']

order_event_files = []
order_PAQ8_files = []
for each in audio_names:
    order_event_files.append(os.path.join(event_path, each))
    order_PAQ8_files.append(os.path.join(source_scene_path, each.replace('_event', '_scene_PAQ')))


data_dict = {}
for num, name in enumerate(audio_names):
    print(num, name)

    sub_dict = {}

    using_index = audio_names.index(name)

    event_pro = np.loadtxt(order_event_files[using_index])
    sub_dict['event'] = event_pro
    sub_dict['event_labels'] = event_labels
    sub_dict['PAQ8ARs'] = emotion

    scenefile = order_PAQ8_files[using_index]
    with open(scenefile, 'r') as f:
        lines = f.readlines()
        sub_dict['scene'] = lines[0].split('\n')[0]
        sub_dict['ISOPle'], sub_dict['ISOEven'] = [float(each) for each in lines[1].split('\n')[0].split('\t')]
        sub_dict['PAQ_8_values'] = [float(each) for each in lines[2].split('\n')[0].split('\t')]

    print(sub_dict)
    data_dict[order_PAQ8_files[using_index].split('\\')[-1]] = sub_dict


output_file = os.path.join(os.getcwd(), 'Dict_data_labels.pickle')
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)

 






