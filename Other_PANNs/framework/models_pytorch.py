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



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x




class PANNs(nn.Module):
    def __init__(self,
                 event_class=len(config.event_labels),
                 scene_class=len(config.scene_labels), batchnormal=False):

        super(PANNs, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

        # # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(2048, event_class, bias=True)
        self.fc_final_scene = nn.Linear(2048, scene_class, bias=True)

        # MSE
        self.fc_final_ISOPls = nn.Linear(2048, 1, bias=True)
        self.fc_final_ISOEvs = nn.Linear(2048, 1, bias=True)

        #
        PAQ_embedding_dim = 2048
        each_emotion_class = 1
        self.fc_final_pleasant = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        # ##############################################################################################################

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input):
        # print(input.shape)

        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # -------------------------------------------------------------------------------------------------------------
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())
        # x.size(): torch.Size([64, 64, 1500, 32])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size()) # x.size(): torch.Size([64, 128, 750, 16])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 256, 375, 8])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 512, 187, 4])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 1024, 93, 2])

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 2048, 93, 2])

        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        # print('6x.size():', x.size())  torch.Size([64, 2048, 93])

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # torch.Size([64, 2048])

        # x = F.dropout(x, p=0.5, training=self.training)
        common_embeddings = F.relu_(self.fc1(x))

        event = self.fc_final_event(common_embeddings)
        scene = self.fc_final_scene(common_embeddings)

        ISOPls = self.fc_final_ISOPls(common_embeddings)
        ISOEvs = self.fc_final_ISOEvs(common_embeddings)

        pleasant = self.fc_final_pleasant(common_embeddings)
        eventful = self.fc_final_eventful(common_embeddings)
        chaotic = self.fc_final_chaotic(common_embeddings)
        vibrant = self.fc_final_vibrant(common_embeddings)
        uneventful = self.fc_final_uneventful(common_embeddings)
        calm = self.fc_final_calm(common_embeddings)
        annoying = self.fc_final_annoying(common_embeddings)
        monotonous = self.fc_final_monotonous(common_embeddings)

        return scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous





