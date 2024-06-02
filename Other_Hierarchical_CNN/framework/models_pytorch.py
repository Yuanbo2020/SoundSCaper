import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_gpu(x, cuda, using_float=False):
    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)





class Hierachical_CNN(nn.Module):
    def __init__(self, batchnormal=True):

        super(Hierachical_CNN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        out_channels = 16
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        out_channels = 32
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv5 = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.conv6 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(2, 0), bias=False)
        self.bn6 = nn.BatchNorm2d(out_channels)

        out_channels = 128
        self.conv7 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(9, 9), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn7 = nn.BatchNorm2d(out_channels)


        each_emotion_class = 1
        d_embeddings = 128
        self.fc_final_event = nn.Linear(d_embeddings, 15, bias=True)
        self.event_embedding_layer = nn.Linear(15, 15, bias=True)
        self.fc_final_scene = nn.Linear(15, 3, bias=True)

        # MSE
        self.PAQ_embedding_layer = nn.Linear(8, 8, bias=True)
        self.fc_final_ISOPls = nn.Linear(8, each_emotion_class, bias=True)
        self.fc_final_ISOEvs = nn.Linear(8, each_emotion_class, bias=True)

        self.fc_final_pleasant = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(d_embeddings, each_emotion_class, bias=True)


    def forward(self, input):
        x = input.unsqueeze(1)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(3, 2))

        x = F.relu_(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=(4, 1))

        x = F.relu_(self.bn7(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=(5, 2))

        x = torch.mean(x, dim=2)

        x = x.view(x.size()[0], -1)

        event = self.fc_final_event(x)
        event_inter = self.event_embedding_layer(event)
        scene = self.fc_final_scene(event_inter)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        PAQ8 = torch.cat(
                [pleasant, eventful, chaotic, vibrant,
                 uneventful, calm, annoying, monotonous], dim=-1)

        PAQ_inter = F.relu_(self.PAQ_embedding_layer(PAQ8))
        ISOPls = self.fc_final_ISOPls(PAQ_inter)
        ISOEvs = self.fc_final_ISOEvs(PAQ_inter)

        return scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous





