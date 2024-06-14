import numpy as np
import h5py, os, pickle, torch
import time
from framework.utilities import calculate_scalar, scale, create_folder
import framework.config as config

import dgl

from dgl.data.utils import save_graphs



class DataGenerator_Mel_loudness_graph(object):
    def __init__(self, Dataset_path, node_emb_dim, number_of_nodes = 8, seed=42, normalization=True, overwrite=False):
        self.Dataset_path = Dataset_path
        self.batch_size = config.batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        ########################################   graph    ###########################################################
        edge_dim = node_emb_dim
        graph_path = os.path.join(config.all_feature_path,
                                  'graph_node' + str(number_of_nodes)
                                  + '_edge_dim_' + str(edge_dim) + '.bin')

        g = dgl.DGLGraph()
        g.add_nodes(number_of_nodes)

        for i in np.arange(number_of_nodes):
            for j in np.arange(number_of_nodes):
                g.add_edges(i, j)

        g.edata['feat'] = torch.ones(g.number_of_edges(), edge_dim)

        self.one_graph = g
        save_graphs(graph_path, [g])
        ################################################################################################################

        file_path = os.path.join(Dataset_path, 'Training_set','training_scene_event_PAQs.pickle')
        all_data = self.load_pickle(file_path)

        self.train_features, self.train_scene_labels, self.train_sound_maskers, self.train_ISOPls, self.train_ISOEvs, \
        self.train_pleasant, self.train_eventful, self.train_chaotic, self.train_vibrant, \
        self.train_uneventful, self.train_calm, self.train_annoying,  self.train_monotonous, _ = self.get_input_output(all_data)

        all_feature_file_path = os.path.join(Dataset_path, 'Training_set', 'training_log_mel.pickle')
        self.train_all_feature_data = self.load_pickle(all_feature_file_path)
        # print(all_feature_data.keys())
        self.train_x = np.array([self.train_all_feature_data[name] for name in self.train_features])
        print('self.train_x: ', self.train_x.shape)
        # self.train_x:  (19152, 3001, 64)

        all_feature_file_path = os.path.join(Dataset_path, 'Training_set', 'training_loudness.pickle')
        self.train_all_feature_data_loudness = self.load_pickle(all_feature_file_path)
        # print(all_feature_data.keys())
        self.train_x_loudness = np.array([self.train_all_feature_data_loudness[name] for name in self.train_features])
        print('self.train_x_loudness: ', self.train_x_loudness.shape)
        # self.train_x_loudness:  (19152, 15000, 1)

        self.normal = normalization
        output_dir = os.path.join(Dataset_path, '0_normalization_files')
        # print('output_dir', output_dir)
        create_folder(output_dir)
        normalization_log_mel_file = os.path.join(output_dir, 'norm_log_mel.pickle')
        normalization_loudness_file = os.path.join(output_dir, 'norm_loudness.pickle')


        if self.normal and not os.path.exists(normalization_loudness_file) or overwrite:
            norm_pickle = {}
            (self.mean_log_mel, self.std_log_mel) = calculate_scalar(np.concatenate(self.train_x))
            norm_pickle['mean'] = self.mean_log_mel
            norm_pickle['std'] = self.std_log_mel
            self.save_pickle(norm_pickle, normalization_log_mel_file)

            norm_pickle = {}
            (self.mean_loudness, self.std_loudness) = calculate_scalar(np.concatenate(self.train_x_loudness))
            norm_pickle['mean'] = self.mean_loudness
            norm_pickle['std'] = self.std_loudness
            self.save_pickle(norm_pickle, normalization_loudness_file)
        else:
            print('using: ', normalization_log_mel_file)
            norm_pickle = self.load_pickle(normalization_log_mel_file)
            self.mean_log_mel = norm_pickle['mean']
            self.std_log_mel = norm_pickle['std']
            print('Log Mel Mean: ', self.mean_log_mel)
            print('Log Mel STD: ', self.std_log_mel)

            print('using: ', normalization_loudness_file)
            norm_pickle = self.load_pickle(normalization_loudness_file)
            self.mean_loudness = norm_pickle['mean']
            self.std_loudness = norm_pickle['std']
            print('Loudness Mean: ', self.mean_loudness)
            print('Loudness STD: ', self.std_loudness)

        print("norm: ", self.mean_log_mel.shape, self.std_log_mel.shape)
        print("norm: ", self.mean_loudness.shape, self.std_loudness.shape)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

    def get_input_output(self, all_data):
        ISOPls, ISOEvs = self.get_ISOPl_ISOEv(all_data)
        scene_labels = self.load_scene_labels(all_data)

        # attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying',  'monotonous']
        audio_names, features, sound_maskers, \
        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying,  monotonous = all_data['soundscape'], all_data['feature_names'], all_data['masker'], \
                                                             all_data['pleasant'], all_data['eventful'], all_data['chaotic'], \
                                                             all_data['vibrant'], all_data['uneventful'], all_data['calm'], \
                                                             all_data['annoying'], all_data['monotonous']

        audio_names = all_data['feature_names']

        assert all_data['all_events'] == config.event_labels
        sound_maskers_labels = all_data['event_labels']

        event_labels = np.zeros((len(sound_maskers_labels), len(config.event_labels)))
        for i, each in enumerate(sound_maskers_labels):
            for sub_each in each:
                event_labels[i, config.event_labels.index(sub_each)] = 1


        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = np.array(pleasant)[:, None], \
                                                                                       np.array(eventful)[:, None], \
                                                                                       np.array(chaotic)[:, None], \
                                                                                       np.array(vibrant)[:, None], \
                                                                                       np.array(uneventful)[:, None], \
                                                                                       np.array(calm)[:, None], \
                                                                                       np.array(annoying)[:, None], \
                                                                                       np.array(monotonous)[:, None]

        # print(pleasant.shape, eventful.shape, chaotic.shape)  # (19152, 1)
        return features, scene_labels, event_labels, ISOPls, ISOEvs, \
               pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying,  monotonous, np.array(audio_names)


    def get_ISOPl_ISOEv(self, all_data):
        attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']  # Define attributes to extract from dataframes
        ISOPl_weights = [1, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0, np.sqrt(2) / 2, -1, -np.sqrt(2) / 2]  # Define weights for each attribute in attributes in computation of ISO Pleasantness
        ISOEv_weights = [0, 1, np.sqrt(2) / 2, np.sqrt(2) / 2, -1, -np.sqrt(2) / 2, 0, -np.sqrt(2) / 2]

        emotion_values = [all_data[each] for each in attributes]

        emotion_values = np.array(emotion_values).transpose((1, 0))
        # print('emotion_values: ', emotion_values.shape)
        # emotion_values:  (19152, 8)

        ISOPls = ((emotion_values * ISOPl_weights).sum(axis=1) / (4 + np.sqrt(32)))


        ISOEvs = ((emotion_values * ISOEv_weights).sum(axis=1) / (4 + np.sqrt(32)))
        ISOPls, ISOEvs = ISOPls[:, None], ISOEvs[:, None]
        return ISOPls, ISOEvs

    def load_scene_labels(self, all_data):
        USotW_acoustic_scene_laebls = all_data['USotW_acoustic_scene_labels']
        clips = all_data['soundscape']
        # print(USotW_acoustic_scene_laebls)
        # print(clips)

        scenes = [USotW_acoustic_scene_laebls[each.split('_44100')[0]] for each in clips]
        correct_scene = []
        for each in scenes:
            if each == 'park ':
                correct_scene.append('park')
            else:
                correct_scene.append(each)

        scene_labels = np.array([config.scene_labels.index(each) for each in correct_scene])

        return scene_labels

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def generate_inference_soundscape_clip_for_LLM(self, Dataset_mel, Dataset_loudness):
        # load
        file_names = []
        self.test_all_feature_data = []
        for file in os.listdir(Dataset_mel):
            file_names.append(file)
            file_path = os.path.join(Dataset_mel, file)
            data = np.load(file_path)
            self.test_all_feature_data.append(data[None, :])

        self.test_all_feature_data_loudness = []
        for file in file_names:
            file_path = os.path.join(Dataset_loudness, file)
            data = np.load(file_path)
            self.test_all_feature_data_loudness.append(data[None, :])

        self.test_x = np.concatenate(self.test_all_feature_data, axis=0)
        print('Inference audio clip mel: ', self.test_x.shape)
        # # self.test_x:  (1, 3001, 64)

        self.test_x_loudness = np.concatenate(self.test_all_feature_data_loudness, axis=0)
        print('Inference audio clip loudness: ', self.test_x_loudness.shape)
        # # self.test_x:  (1, 3001, 64)

        audios_num = len(self.test_x)

        audio_indexes = [i for i in range(audios_num)]

        print('Number of {} audio clip(s) in inference'.format(len(self.test_x)))

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            # ---
            batch_x = self.test_x[batch_audio_indexes]
            batch_x_loudness = self.test_x_loudness[batch_audio_indexes]
            if self.normal:
                batch_x = self.transform(batch_x, self.mean_log_mel, self.std_log_mel)
                batch_x_loudness = self.transform(batch_x_loudness, self.mean_loudness, self.std_loudness)

            batch_graph = [self.one_graph for j in range(self.batch_size)]

            yield batch_x, batch_x_loudness, batch_graph, file_names


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, mean, std)





