import sys, os, argparse

gpu_id = 1
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
    file = 'SoundAQnet_ASC96_AEC94_PAQ1027.pth'
    # Parameters num: 2.701812 M
    # ASC	Acc:  96.09 %
    # AEC	AUC:  0.94
    # PAQ_8D_AQ	MSE MEAN: 1.027
    # pleasant_mse: 0.880 eventful_mse: 1.029 chaotic_mse: 1.043 vibrant_mse: 0.972
    # uneventful_mse: 1.126 calm_mse: 0.969 annoying_mse: 1.055 monotonous_mse: 1.140

    # file = 'SoundAQnet_ASC96_AEC94_PAQ1039.pth'
    # # ASC	Acc:  95.69 %
    # # AEC	AUC:  0.94
    # # PAQ_8D_AQ	MSE MEAN: 1.039
    # # pleasant_mse: 0.887 eventful_mse: 1.041 chaotic_mse: 1.062 vibrant_mse: 0.979
    # # uneventful_mse: 1.152 calm_mse: 0.981 annoying_mse: 1.058 monotonous_mse: 1.154

    # file = 'SoundAQnet_ASC96_AEC95_PAQ1052.pth'
    # # ASC	Acc:  95.52 %
    # # AEC	AUC:  0.95
    # # PAQ_8D_AQ	MSE MEAN: 1.052
    # # pleasant_mse: 0.902 eventful_mse: 1.057 chaotic_mse: 1.073 vibrant_mse: 0.987
    # # uneventful_mse: 1.157 calm_mse: 1.006 annoying_mse: 1.068 monotonous_mse: 1.167

    # file = 'SoundAQnet_ASC96_AEC94_PAQ1041.pth'
    # # ASC	Acc:  95.99 %
    # # AEC	AUC:  0.94
    # # PAQ_8D_AQ	MSE MEAN: 1.041
    # # pleasant_mse: 0.890 eventful_mse: 1.047 chaotic_mse: 1.052 vibrant_mse: 0.974
    # # uneventful_mse: 1.152 calm_mse: 0.995 annoying_mse: 1.065 monotonous_mse: 1.151

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
    generate_func = generator.generate_testing(data_type='testing')
    dict = forward(model=model, generate_func=generate_func, cuda=config.cuda)

    # AUC
    event_auc = cal_auc(dict['label_event'], dict['output_event'])

    # softmax classification acc
    scene_acc = cal_softmax_classification_accuracy(dict['label_scene'], dict['output_scene'], average='macro')

    pleasant_mse = metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant'])
    eventful_mse = metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful'])
    chaotic_mse = metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic'])
    vibrant_mse = metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant'])
    uneventful_mse = metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful'])
    calm_mse = metrics.mean_squared_error(dict['label_calm'], dict['output_calm'])
    annoying_mse = metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying'])
    monotonous_mse = metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous'])

    PAQ8 = [pleasant_mse, eventful_mse, chaotic_mse, vibrant_mse, uneventful_mse, calm_mse, annoying_mse, monotonous_mse]

    PAQ8_mean = np.mean(PAQ8)

    # print('ASC\tAcc: ', "%.2f"%(scene_acc*100), '%')
    # print(f'AEC\tAUC: {"%.2f"%(event_auc)}')
    # print(f'PAQ_8D_AQ\tMEAN MSE: {"%.3f"%(PAQ8_mean)}')

    params_num = count_parameters(model)
    print('Parameters num: {} M'.format(params_num / 1000 ** 2))

    print('ASC\tAcc: ', "%.2f" % (scene_acc * 100), '%')
    print('AEC\tAUC: ', "%.2f" % (event_auc))
    print(f'PAQ_8D_AQ\tMSE MEAN: {"%.3f" % (PAQ8_mean)}')

    print('pleasant_mse: %.3f' % float(pleasant_mse), 'eventful_mse: %.3f' % float(eventful_mse),
          'chaotic_mse: %.3f' % float(chaotic_mse), 'vibrant_mse: %.3f' % float(vibrant_mse))
    print('uneventful_mse: %.3f' % float(uneventful_mse), 'calm_mse: %.3f' % float(calm_mse),
          'annoying_mse: %.3f' % float(annoying_mse), 'monotonous_mse: %.3f' % float(monotonous_mse))





if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















