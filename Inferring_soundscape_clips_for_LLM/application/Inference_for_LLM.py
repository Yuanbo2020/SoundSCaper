import sys, os, argparse

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.processing import *
from framework.data_generator import *
from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters


def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def main(argv): 

    node_emb_dim = 64
    hidden_dim, out_dim = 32, 64
    batch_size = 32
    number_of_nodes = 8
    monitor = 'ISOPls'

    using_model = SoundAQnet

    model = using_model(
        max_node_num=number_of_nodes,
        node_emb_dim=node_emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim)

    print(model)

    syspath = os.path.join(os.getcwd(), 'system', 'model')
    file = 'SoundAQnet_PAQ1075.pth'
    file = 'SoundAQnet_PAQ1054.pth'

    event_model_path = os.path.join(syspath, file)

    model_event = torch.load(event_model_path, map_location='cpu')

    if 'state_dict' in model_event.keys():
        model.load_state_dict(model_event['state_dict'])
    else:
        model.load_state_dict(model_event)

    if config.cuda:
        model.cuda()

    Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_Mel_loudness_graph(Dataset_path, node_emb_dim, number_of_nodes)

    # Generate function
    Dataset_mel = os.path.join(os.getcwd(), 'Dataset_mel')
    Dataset_loudness = os.path.join(os.getcwd(), 'Dataset_wav_loudness')
    generate_func = generator.generate_inference_soundscape_clip_for_LLM(Dataset_mel, Dataset_loudness)
    dict = forward_for_LLM(model=model, generate_func=generate_func, cuda=config.cuda)

    ###################################################################################################################

    result_event_dir = os.path.join(os.getcwd(), 'SoundAQnet_event_probability')
    create_folder(result_event_dir)

    result_PAQ_dir = os.path.join(os.getcwd(), 'SoundAQnet_scene_ISOPl_ISOEv_PAQ8DARs')
    create_folder(result_PAQ_dir)

    for each_index, name in enumerate(dict['audio_names']):
        # print(each_index, name)
        # 0 fold_1_participant_00056_stimulus_13.npy

        print('Soundscape audio clip:', name.split('.npy')[0])

        txtfile = os.path.join(result_event_dir, name.replace('.npy', '_event.txt'))
        # print(txtfile)
        np.savetxt(txtfile, dict['output_event'][each_index])
        print('Audio event probability matrix:', dict['output_event'][each_index])

        txtfile = os.path.join(result_PAQ_dir, name.replace('.npy', '_scene_PAQ.txt'))
        # print(txtfile)

        with open(txtfile, 'w') as f:
            max_id = np.argmax(dict['output_scene'][each_index])
            print('Acoustic scene:', config.scene_labels[max_id])
            f.write(config.scene_labels[max_id] + '\n')
            f.write(str(dict['output_ISOPls'][each_index][0]) + '\t' + str(dict['output_ISOEvs'][each_index][0]) + '\n')
            print('ISOP and ISOE:', str(dict['output_ISOPls'][each_index][0]) + '\t' + str(dict['output_ISOEvs'][each_index][0]))

            print('PAQ 8D ARs:', str(dict['output_pleasant'][each_index][0]) + '\t' +
                    str(dict['output_eventful'][each_index][0]) + '\t' +
                    str(dict['output_chaotic'][each_index][0]) + '\t' +
                    str(dict['output_vibrant'][each_index][0]) + '\t' +
                    str(dict['output_uneventful'][each_index][0]) + '\t' +
                    str(dict['output_calm'][each_index][0]) + '\t' +
                    str(dict['output_annoying'][each_index][0]) + '\t' +
                    str(dict['output_monotonous'][each_index][0]))

            f.write(str(dict['output_pleasant'][each_index][0]) + '\t' +
                    str(dict['output_eventful'][each_index][0]) + '\t' +
                    str(dict['output_chaotic'][each_index][0]) + '\t' +
                    str(dict['output_vibrant'][each_index][0]) + '\t' +
                    str(dict['output_uneventful'][each_index][0]) + '\t' +
                    str(dict['output_calm'][each_index][0]) + '\t' +
                    str(dict['output_annoying'][each_index][0]) + '\t' +
                    str(dict['output_monotonous'][each_index][0]) + '\n')

        print('\n')





if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















