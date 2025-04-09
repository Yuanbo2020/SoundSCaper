import random
import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *


def main(argv):
    node_emb_dim = 64
    hidden_dim, out_dim = 32, 64
    batch_size = 32
    number_of_nodes = 8
    monitor = 'ISOPls'

    sys_name = 'system'
    system_path = os.path.join(os.getcwd(), sys_name)
    models_dir = os.path.join(system_path, 'model')

    using_model = SoundAQnet_block2

    model = using_model(
        max_node_num=number_of_nodes,
        node_emb_dim=node_emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        batchnormal=True)

    if config.cuda and torch.cuda.is_available():
        model.cuda()

    Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_Mel_loudness_graph(Dataset_path, node_emb_dim, number_of_nodes = number_of_nodes)

    Training_early_stopping(generator, model, models_dir, batch_size, monitor)

    print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















