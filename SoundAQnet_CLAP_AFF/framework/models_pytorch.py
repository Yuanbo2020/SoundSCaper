import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config
import dgl
from framework.gated_gcn_layer import GatedGCNLayer


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



# ----------------------------------------------------------------------------------------------------------------------

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



class AFF(nn.Module):
    '''
    Attentional Feature Fusion
    '''

    def __init__(self, channels=64, r=4, type='2D'):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        if type == '1D':
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == '2D':
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f'the type is not supported.'

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class SoundAQnet_CLAP_AFF_fusion(nn.Module):
    def __init__(self, max_node_num, node_emb_dim = 256,
        hidden_dim = 32,
        out_dim = 64,
        n_layers = 1,
                 event_class=len(config.event_labels),
                 scene_class=len(config.scene_labels), each_emotion_class=config.each_emotion_class_num, batchnormal=True):

        super(SoundAQnet_CLAP_AFF_fusion, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 5
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock_dilation(in_channels=16, out_channels=32, padding=(0,0), dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation(in_channels=32, out_channels=64, padding=(0,0), dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0,2))
        self.conv_block2_kernel_5 = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 9
        kernel_size = (9, 9)
        self.conv_block1_kernel_9 = ConvBlock(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 4))
        self.conv_block2_kernel_9 = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 3), dilation=(2, 1))
        self.conv_block3_kernel_9 = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 3), dilation=(3, 1))
        self.k_9_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        self.clap_fusion_mel = AFF(channels=64, type='2D')

        # ---------------- loudness -----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1_loudness = ConvBlock(kernel_size=(3, 1), padding=(0, 0), in_channels=1, out_channels=16)
        self.conv_block2_loudness = ConvBlock_dilation(kernel_size=(3, 1), in_channels=16, out_channels=32, padding=(0, 0),
                                              dilation=(2, 1))
        self.conv_block3_loudness = ConvBlock_dilation(kernel_size=(3, 1), in_channels=32, out_channels=64, padding=(0, 0),
                                              dilation=(3, 1))

        # -------------- kernel 5
        kernel_size = (5, 1)
        self.conv_block1_kernel_5_loudness = ConvBlock(in_channels=1, out_channels=16,
                                              kernel_size=kernel_size, padding=(0, 0))
        self.conv_block2_kernel_5_loudness = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(2, 1))
        self.conv_block3_kernel_5_loudness = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(3, 1))

        # -------------- kernel 7
        kernel_size = (7, 1)
        self.conv_block1_kernel_7_loudness = ConvBlock(in_channels=1, out_channels=16,
                                              kernel_size=kernel_size, padding=(0, 0))
        self.conv_block2_kernel_7_loudness = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(2, 1))
        self.conv_block3_kernel_7_loudness = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(3, 1))

        # -------------- kernel 9
        kernel_size = (9, 1)
        self.conv_block1_kernel_9_loudness = ConvBlock(in_channels=1, out_channels=16,
                                              kernel_size=kernel_size, padding=(0, 0))
        self.conv_block2_kernel_9_loudness = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(2, 1))
        self.conv_block3_kernel_9_loudness = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(3, 1))

        # --------------------------------------------------------------------------------------------------------------
        self.clap_fusion_loudness = AFF(channels=64, type='2D')

        self.clap_fusion_mel_loudness = AFF(channels=64, type='2D')
        ##################################### gnn ####################################################################

        in_dim = node_emb_dim
        in_dim_edge = in_dim

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.max_node_num = max_node_num

        # ----------------------------------------------------------------------------------------------------


        graph_dim = 512
        self.fc_all_nodes_to_classification_embeddings = nn.Linear(64 * 8, graph_dim, bias=True)

        self.fc_residual_mel_k3 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_mel_k5 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_mel_k7 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_mel_k9 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)

        self.fc_residual_loudness_k3 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_loudness_k5 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_loudness_k7 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_loudness_k9 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)

        scene_event_embedding_dim = 256
        # embedding layers
        self.fc_embedding_event = nn.Linear(graph_dim, scene_event_embedding_dim, bias=True)
        self.fc_embedding_scene = nn.Linear(graph_dim, scene_event_embedding_dim, bias=True)

        # MSE
        ISO_affective_embedding_dim = 64
        self.fc_embedding_ISOPls = nn.Linear(graph_dim, ISO_affective_embedding_dim, bias=True)
        self.fc_embedding_ISOEvs = nn.Linear(graph_dim, ISO_affective_embedding_dim, bias=True)

        PAQ_embedding_dim = 128
        self.fc_embedding_pleasant = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_eventful = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_chaotic = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_vibrant = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_uneventful = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_calm = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_annoying = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_monotonous = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        # -----------------------------------------------------------------------------------------------------------

        # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(scene_event_embedding_dim, event_class, bias=True)
        self.fc_final_scene = nn.Linear(scene_event_embedding_dim, scene_class, bias=True)

        # MSE
        self.fc_final_ISOPls = nn.Linear(ISO_affective_embedding_dim, 1, bias=True)
        self.fc_final_ISOEvs = nn.Linear(ISO_affective_embedding_dim, 1, bias=True)

        self.fc_final_pleasant = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        ##############################################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc_all_nodes_to_classification_embeddings)

        init_layer(self.fc_embedding_event)
        init_layer(self.fc_embedding_scene)

        init_layer(self.fc_embedding_ISOPls)
        init_layer(self.fc_embedding_ISOEvs)

        init_layer(self.fc_embedding_pleasant)
        init_layer(self.fc_embedding_eventful)
        init_layer(self.fc_embedding_chaotic)
        init_layer(self.fc_embedding_vibrant)
        init_layer(self.fc_embedding_uneventful)
        init_layer(self.fc_embedding_calm)
        init_layer(self.fc_embedding_annoying)
        init_layer(self.fc_embedding_monotonous)

        # classification layer -------------------------------------------------------------------------------------
        init_layer(self.fc_final_event)
        init_layer(self.fc_final_scene)

        init_layer(self.fc_final_ISOPls)
        init_layer(self.fc_final_ISOEvs)

        init_layer(self.fc_final_pleasant)
        init_layer(self.fc_final_eventful)
        init_layer(self.fc_final_chaotic)
        init_layer(self.fc_final_vibrant)
        init_layer(self.fc_final_uneventful)
        init_layer(self.fc_final_calm)
        init_layer(self.fc_final_annoying)
        init_layer(self.fc_final_monotonous)

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input, batch_x_loudness, batch_graph):
        # print(input.shape)

        # torch.Size([32, 3001, 64])
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # # torch.Size([32, 15000, 1])
        (_, seq_len_loudness, mel_bins_loudness) = batch_x_loudness.shape
        batch_x_loudness = batch_x_loudness.view(-1, 1, seq_len_loudness, mel_bins_loudness)

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        batch_x = x

        # print(x.size())  # torch.Size([32, 1, 3001, 64])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block2(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size())

        x_k_3 = self.conv_block3(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size())  torch.Size([32, 64, 367, 5])

        x_k_3 = self.mean_max(x_k_3)
        # print(x_k_3.size()) torch.Size([32, 64, 5])
        # x_k_3_mel = F.relu_(self.k_3_freq_to_1(x_k_3))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])


        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 64, 1496, 64])

        x_k_5 = self.conv_block2_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 128, 740, 52])

        x_k_5 = self.conv_block3_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([8, 256, 358, 32])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        # x_k_5_mel = F.relu_(self.k_5_freq_to_1(x_k_5))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 64, 1494, 64])

        x_k_7 = self.conv_block2_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 128, 735, 48])

        x_k_7 = self.conv_block3_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([8, 256, 349, 20])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        # x_k_7_mel = F.relu_(self.k_7_freq_to_1(x_k_7))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 9 -----------------------------------------------------------------------------------------------------
        x_k_9 = self.conv_block1_kernel_9(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([8, 64, 1492, 64])

        x_k_9 = self.conv_block2_kernel_9(x_k_9, pool_size=(2, 2), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([8, 128, 730, 44])

        x_k_9 = self.conv_block3_kernel_9(x_k_9, pool_size=(2, 2), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size(), '\n')  # torch.Size([8, 256, 341, 8])

        x_k_9 = self.mean_max(x_k_9)  # torch.Size([8, 256, 5])
        # x_k_9_mel = F.relu_(self.k_9_freq_to_1(x_k_9))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])
        # torch.Size([1, 8, 256])

        # -------------------------------------------------------------------------------------------------------------
        # event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel, x_k_7_mel, x_k_9_mel], dim=0)
        event_embs_log_mel = torch.cat([x_k_3[None, :, :, :], x_k_5[None, :, :, :],
                                        x_k_7[None, :, :, :], x_k_9[None, :, :, :]], dim=0)
        # print(event_embs_log_mel.size())  # torch.Size([4, 16, 64, 5])  (node_num, batch, edge_dim)
        # torch.Size([4, 32, 64, 5])

        global_mel = torch.mean(event_embs_log_mel, dim=0)[:, :, :, None]  # torch.Size([32, 64, 5, 1])

        # print(x_k_3_mel.size(), x_k_5_mel.size(), x_k_7_mel.size(), x_k_9_mel.size())
        # torch.Size([1, 32, 64]) torch.Size([1, 32, 64]) torch.Size([1, 32, 64]) torch.Size([1, 32, 64])

        x_k_3_mel = self.clap_fusion_mel(global_mel, x_k_3[:, :, :, None])
        # print(x_k_3_mel.size())  torch.Size([32, 64, 5, 1])
        x_k_5_mel = self.clap_fusion_mel(global_mel, x_k_5[:, :, :, None])
        x_k_7_mel = self.clap_fusion_mel(global_mel, x_k_7[:, :, :, None])
        x_k_9_mel = self.clap_fusion_mel(global_mel, x_k_9[:, :, :, None])

        x_overall_mel = torch.cat([x_k_3_mel, x_k_5_mel, x_k_7_mel, x_k_9_mel], dim=-1)
        # print(x_overall_mel.size())
        #  torch.Size([32, 64, 5, 4])
        # x_overall_mel = x_overall_mel.flatten(1)

        #  ----------------------------- loudness ----------------------------------------------------------------------
        batch_x = batch_x_loudness

        # print(x.size())  # torch.Size([64, 1, 15000, 1])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size())  # torch.Size([64, 16, 7498, 1])

        x_k_3 = self.conv_block2_loudness(x_k_3, pool_size=(2, 1), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size())  # torch.Size([64, 32, 3745, 1])

        x_k_3 = self.conv_block3_loudness(x_k_3, pool_size=(2, 1), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size(), '\n')  # torch.Size([64, 64, 1866, 1])

        x_k_3 = self.mean_max(x_k_3)  # torch.Size([8, 64, 1])
        # x_k_3_loudness = x_k_3[:, :, 0][None, :, :]  # torch.Size([8, 64, 1])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([64, 16, 7496, 1])

        x_k_5 = self.conv_block2_kernel_5_loudness(x_k_5, pool_size=(2, 1), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([64, 32, 3740, 1])

        x_k_5 = self.conv_block3_kernel_5_loudness(x_k_5, pool_size=(2, 1), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([64, 64, 1858, 1])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        # x_k_5_loudness = x_k_5[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([64, 32, 7494, 1])

        x_k_7 = self.conv_block2_kernel_7_loudness(x_k_7, pool_size=(2, 1), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([64, 64, 3735, 1])

        x_k_7 = self.conv_block3_kernel_7_loudness(x_k_7, pool_size=(2, 1), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([64, 64, 1849, 1])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        # x_k_7_loudness = x_k_7[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 9 -----------------------------------------------------------------------------------------------------
        x_k_9 = self.conv_block1_kernel_9_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([64, 16, 7492, 1])

        x_k_9 = self.conv_block2_kernel_9_loudness(x_k_9, pool_size=(2, 1), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([64, 32, 3730, 1])

        x_k_9 = self.conv_block3_kernel_9_loudness(x_k_9, pool_size=(2, 1), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size(), '\n')  # torch.Size([64, 64, 1841, 1])

        x_k_9 = self.mean_max(x_k_9)  # torch.Size([8, 256, 5])
        # x_k_9_loudness = x_k_9[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # -------------------------------------------------------------------------------------------------------------
        # event_embs_loudness = torch.cat([x_k_3_loudness, x_k_5_loudness, x_k_7_loudness, x_k_9_loudness], dim=0)

        event_embs_loudness = torch.cat([x_k_3[None, :, :, :], x_k_5[None, :, :, :],
                                        x_k_7[None, :, :, :], x_k_9[None, :, :, :]], dim=0)
        # print(event_embs_loudness.size())  # torch.Size([4, 16, 64, 5])  (node_num, batch, edge_dim)
        # torch.Size([4, 32, 64, 1])

        global_loudness = torch.mean(event_embs_loudness, dim=0)[:, :, :, None]  # torch.Size([32, 64, 5, 1])

        x_k_3_loudness = self.clap_fusion_loudness(global_loudness, x_k_3[:, :, :, None])
        # print(x_k_3_loudness.size())  torch.Size([32, 64, 5, 1])
        x_k_5_loudness = self.clap_fusion_loudness(global_loudness, x_k_5[:, :, :, None])
        x_k_7_loudness = self.clap_fusion_loudness(global_loudness, x_k_7[:, :, :, None])
        x_k_9_loudness = self.clap_fusion_loudness(global_loudness, x_k_9[:, :, :, None])

        x_overall_loudness = torch.cat([x_k_3_loudness, x_k_5_loudness, x_k_7_loudness, x_k_9_loudness], dim=-1)
        # print(x_overall_loudness.size()) torch.Size([32, 64, 1, 4])

        mel_loudness = torch.cat([x_overall_mel, x_overall_loudness], dim=2)
        mel_loudness = torch.mean(mel_loudness, dim=2)[:, :, :, None]
        # print(mel_loudness.size())

        # print(mel_loudness.size(), x_overall_mel.size()) torch.Size([32, 64, 4, 1]) torch.Size([32, 64, 5, 4])
        x_overall_mel = self.clap_fusion_mel_loudness(mel_loudness, torch.mean(x_overall_mel, dim=2)[:, :, :, None])
        # print(x_overall_mel.size())

        # print(mel_loudness.size(), x_overall_loudness.size())
        x_overall_loudness = self.clap_fusion_mel_loudness(mel_loudness, torch.mean(x_overall_loudness, dim=2)[:, :, :, None])
        # print(x_overall_loudness.size()) torch.Size([32, 64, 4, 1])

        x_overall = torch.cat([x_overall_mel, x_overall_loudness], dim=-1)

        kernels_embs = x_overall.flatten(1)

        common_embeddings = F.gelu(self.fc_all_nodes_to_classification_embeddings(kernels_embs))

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.gelu(self.fc_embedding_event(common_embeddings))
        scene_embeddings = F.gelu(self.fc_embedding_scene(common_embeddings))

        ISOPls_embeddings = F.gelu(self.fc_embedding_ISOPls(common_embeddings))
        ISOEvs_embeddings = F.gelu(self.fc_embedding_ISOEvs(common_embeddings))

        pleasant_embeddings = F.gelu(self.fc_embedding_pleasant(common_embeddings))
        eventful_embeddings = F.gelu(self.fc_embedding_eventful(common_embeddings))
        chaotic_embeddings = F.gelu(self.fc_embedding_chaotic(common_embeddings))
        vibrant_embeddings = F.gelu(self.fc_embedding_vibrant(common_embeddings))
        uneventful_embeddings = F.gelu(self.fc_embedding_uneventful(common_embeddings))
        calm_embeddings = F.gelu(self.fc_embedding_calm(common_embeddings))
        annoying_embeddings = F.gelu(self.fc_embedding_annoying(common_embeddings))
        monotonous_embeddings = F.gelu(self.fc_embedding_monotonous(common_embeddings))
        # -------------------------------------------------------------------------------------------------------------

        event = self.fc_final_event(event_embeddings)
        scene = self.fc_final_scene(scene_embeddings)

        ISOPls = self.fc_final_ISOPls(ISOPls_embeddings)
        ISOEvs = self.fc_final_ISOEvs(ISOEvs_embeddings)

        pleasant = self.fc_final_pleasant(pleasant_embeddings)
        eventful = self.fc_final_eventful(eventful_embeddings)
        chaotic = self.fc_final_chaotic(chaotic_embeddings)
        vibrant = self.fc_final_vibrant(vibrant_embeddings)
        uneventful = self.fc_final_uneventful(uneventful_embeddings)
        calm = self.fc_final_calm(calm_embeddings)
        annoying = self.fc_final_annoying(annoying_embeddings)
        monotonous = self.fc_final_monotonous(monotonous_embeddings)

        return scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous


class SoundAQnet(nn.Module):
    def __init__(self, max_node_num, node_emb_dim = 256,
        hidden_dim = 32,
        out_dim = 64,
        n_layers = 1,
                 event_class=len(config.event_labels),
                 scene_class=len(config.scene_labels), each_emotion_class=config.each_emotion_class_num, batchnormal=True):

        super(SoundAQnet, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 5
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock_dilation(in_channels=16, out_channels=32, padding=(0,0), dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation(in_channels=32, out_channels=64, padding=(0,0), dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0,2))
        self.conv_block2_kernel_5 = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 9
        kernel_size = (9, 9)
        self.conv_block1_kernel_9 = ConvBlock(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 4))
        self.conv_block2_kernel_9 = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 3), dilation=(2, 1))
        self.conv_block3_kernel_9 = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 3), dilation=(3, 1))
        self.k_9_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)


        # ---------------- loudness -----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1_loudness = ConvBlock(kernel_size=(3, 1), padding=(0, 0), in_channels=1, out_channels=16)
        self.conv_block2_loudness = ConvBlock_dilation(kernel_size=(3, 1), in_channels=16, out_channels=32, padding=(0, 0),
                                              dilation=(2, 1))
        self.conv_block3_loudness = ConvBlock_dilation(kernel_size=(3, 1), in_channels=32, out_channels=64, padding=(0, 0),
                                              dilation=(3, 1))

        # -------------- kernel 5
        kernel_size = (5, 1)
        self.conv_block1_kernel_5_loudness = ConvBlock(in_channels=1, out_channels=16,
                                              kernel_size=kernel_size, padding=(0, 0))
        self.conv_block2_kernel_5_loudness = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(2, 1))
        self.conv_block3_kernel_5_loudness = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(3, 1))

        # -------------- kernel 7
        kernel_size = (7, 1)
        self.conv_block1_kernel_7_loudness = ConvBlock(in_channels=1, out_channels=16,
                                              kernel_size=kernel_size, padding=(0, 0))
        self.conv_block2_kernel_7_loudness = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(2, 1))
        self.conv_block3_kernel_7_loudness = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(3, 1))

        # -------------- kernel 9
        kernel_size = (9, 1)
        self.conv_block1_kernel_9_loudness = ConvBlock(in_channels=1, out_channels=16,
                                              kernel_size=kernel_size, padding=(0, 0))
        self.conv_block2_kernel_9_loudness = ConvBlock_dilation(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(2, 1))
        self.conv_block3_kernel_9_loudness = ConvBlock_dilation(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 0), dilation=(3, 1))

        # --------------------------------------------------------------------------------------------------------------

        ##################################### gnn ####################################################################

        in_dim = node_emb_dim
        in_dim_edge = in_dim

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.max_node_num = max_node_num

        # ----------------------------------------------------------------------------------------------------


        graph_dim = 512
        self.fc_all_nodes_to_classification_embeddings = nn.Linear(max_node_num * node_emb_dim, graph_dim, bias=True)

        self.fc_residual_mel_k3 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_mel_k5 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_mel_k7 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_mel_k9 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)

        self.fc_residual_loudness_k3 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_loudness_k5 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_loudness_k7 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)
        self.fc_residual_loudness_k9 = nn.Linear(2 * node_emb_dim, node_emb_dim, bias=True)

        scene_event_embedding_dim = 256
        # embedding layers
        self.fc_embedding_event = nn.Linear(graph_dim, scene_event_embedding_dim, bias=True)
        self.fc_embedding_scene = nn.Linear(graph_dim, scene_event_embedding_dim, bias=True)

        # MSE
        ISO_affective_embedding_dim = 64
        self.fc_embedding_ISOPls = nn.Linear(graph_dim, ISO_affective_embedding_dim, bias=True)
        self.fc_embedding_ISOEvs = nn.Linear(graph_dim, ISO_affective_embedding_dim, bias=True)

        PAQ_embedding_dim = 128
        self.fc_embedding_pleasant = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_eventful = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_chaotic = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_vibrant = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_uneventful = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_calm = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_annoying = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        self.fc_embedding_monotonous = nn.Linear(graph_dim, PAQ_embedding_dim, bias=True)
        # -----------------------------------------------------------------------------------------------------------

        # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(scene_event_embedding_dim, event_class, bias=True)
        self.fc_final_scene = nn.Linear(scene_event_embedding_dim, scene_class, bias=True)

        # MSE
        self.fc_final_ISOPls = nn.Linear(ISO_affective_embedding_dim, 1, bias=True)
        self.fc_final_ISOEvs = nn.Linear(ISO_affective_embedding_dim, 1, bias=True)

        self.fc_final_pleasant = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(PAQ_embedding_dim, each_emotion_class, bias=True)
        ##############################################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc_all_nodes_to_classification_embeddings)

        init_layer(self.fc_embedding_event)
        init_layer(self.fc_embedding_scene)

        init_layer(self.fc_embedding_ISOPls)
        init_layer(self.fc_embedding_ISOEvs)

        init_layer(self.fc_embedding_pleasant)
        init_layer(self.fc_embedding_eventful)
        init_layer(self.fc_embedding_chaotic)
        init_layer(self.fc_embedding_vibrant)
        init_layer(self.fc_embedding_uneventful)
        init_layer(self.fc_embedding_calm)
        init_layer(self.fc_embedding_annoying)
        init_layer(self.fc_embedding_monotonous)

        # classification layer -------------------------------------------------------------------------------------
        init_layer(self.fc_final_event)
        init_layer(self.fc_final_scene)

        init_layer(self.fc_final_ISOPls)
        init_layer(self.fc_final_ISOEvs)

        init_layer(self.fc_final_pleasant)
        init_layer(self.fc_final_eventful)
        init_layer(self.fc_final_chaotic)
        init_layer(self.fc_final_vibrant)
        init_layer(self.fc_final_uneventful)
        init_layer(self.fc_final_calm)
        init_layer(self.fc_final_annoying)
        init_layer(self.fc_final_monotonous)

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input, batch_x_loudness, batch_graph):
        # print(input.shape)

        # torch.Size([32, 3001, 64])
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # # torch.Size([32, 15000, 1])
        (_, seq_len_loudness, mel_bins_loudness) = batch_x_loudness.shape
        batch_x_loudness = batch_x_loudness.view(-1, 1, seq_len_loudness, mel_bins_loudness)

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        batch_x = x

        # print(x.size())  # torch.Size([32, 1, 3001, 64])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block2(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block3(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.mean_max(x_k_3)
        x_k_3_mel = F.relu_(self.k_3_freq_to_1(x_k_3))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 64, 1496, 64])

        x_k_5 = self.conv_block2_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 128, 740, 52])

        x_k_5 = self.conv_block3_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([8, 256, 358, 32])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        x_k_5_mel = F.relu_(self.k_5_freq_to_1(x_k_5))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 64, 1494, 64])

        x_k_7 = self.conv_block2_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 128, 735, 48])

        x_k_7 = self.conv_block3_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([8, 256, 349, 20])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        x_k_7_mel = F.relu_(self.k_7_freq_to_1(x_k_7))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 9 -----------------------------------------------------------------------------------------------------
        x_k_9 = self.conv_block1_kernel_9(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([8, 64, 1492, 64])

        x_k_9 = self.conv_block2_kernel_9(x_k_9, pool_size=(2, 2), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([8, 128, 730, 44])

        x_k_9 = self.conv_block3_kernel_9(x_k_9, pool_size=(2, 2), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size(), '\n')  # torch.Size([8, 256, 341, 8])

        x_k_9 = self.mean_max(x_k_9)  # torch.Size([8, 256, 5])
        x_k_9_mel = F.relu_(self.k_9_freq_to_1(x_k_9))[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])
        # torch.Size([1, 8, 256])

        # -------------------------------------------------------------------------------------------------------------
        event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel, x_k_7_mel, x_k_9_mel], dim=0)
        # print(event_embs_log_mel.size())  # torch.Size([4, 16, 64])  (node_num, batch, edge_dim)

        #  ----------------------------- loudness ----------------------------------------------------------------------
        batch_x = batch_x_loudness

        # print(x.size())  # torch.Size([64, 1, 15000, 1])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size())  # torch.Size([64, 16, 7498, 1])

        x_k_3 = self.conv_block2_loudness(x_k_3, pool_size=(2, 1), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size())  # torch.Size([64, 32, 3745, 1])

        x_k_3 = self.conv_block3_loudness(x_k_3, pool_size=(2, 1), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print(x_k_3.size(), '\n')  # torch.Size([64, 64, 1866, 1])

        x_k_3 = self.mean_max(x_k_3)  # torch.Size([8, 64, 1])
        x_k_3_loudness = x_k_3[:, :, 0][None, :, :]  # torch.Size([8, 64, 1])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([64, 16, 7496, 1])

        x_k_5 = self.conv_block2_kernel_5_loudness(x_k_5, pool_size=(2, 1), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([64, 32, 3740, 1])

        x_k_5 = self.conv_block3_kernel_5_loudness(x_k_5, pool_size=(2, 1), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([64, 64, 1858, 1])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        x_k_5_loudness = x_k_5[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([64, 32, 7494, 1])

        x_k_7 = self.conv_block2_kernel_7_loudness(x_k_7, pool_size=(2, 1), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([64, 64, 3735, 1])

        x_k_7 = self.conv_block3_kernel_7_loudness(x_k_7, pool_size=(2, 1), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([64, 64, 1849, 1])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        x_k_7_loudness = x_k_7[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # kernel 9 -----------------------------------------------------------------------------------------------------
        x_k_9 = self.conv_block1_kernel_9_loudness(batch_x, pool_size=(2, 1), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([64, 16, 7492, 1])

        x_k_9 = self.conv_block2_kernel_9_loudness(x_k_9, pool_size=(2, 1), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size())  # torch.Size([64, 32, 3730, 1])

        x_k_9 = self.conv_block3_kernel_9_loudness(x_k_9, pool_size=(2, 1), pool_type='avg')
        x_k_9 = F.dropout(x_k_9, p=0.2, training=self.training)
        # print(x_k_9.size(), '\n')  # torch.Size([64, 64, 1841, 1])

        x_k_9 = self.mean_max(x_k_9)  # torch.Size([8, 256, 5])
        x_k_9_loudness = x_k_9[:, :, 0][None, :, :]  # torch.Size([8, 256, 1])

        # -------------------------------------------------------------------------------------------------------------
        event_embs_loudness = torch.cat([x_k_3_loudness, x_k_5_loudness, x_k_7_loudness, x_k_9_loudness], dim=0)

        event_embs = torch.cat([event_embs_log_mel, event_embs_loudness], dim=0)
        ##################################### gnn ####################################################################

        batched_graph = []  # dgl.batch(batch_x)
        for each_num in range(event_embs.size()[1]):
            h = event_embs[:, each_num, :]
            g = batch_graph[each_num].to('cuda:0')
            g.ndata['feat'] = h
            batched_graph.append(g)

        batched_graph = dgl.batch(batched_graph)
        batch_edges = batched_graph.edata['feat']
        batch_nodes = batched_graph.ndata['feat']

        h = self.embedding_h(batch_nodes)
        e = self.embedding_e(batch_edges)

        # convnets
        for conv in self.layers:
            h, e, mini_graph = conv(batched_graph, h, e)

        x = h.view(-1, self.max_node_num, self.out_dim)  # batch, num, dim
        ######################################## event graph ##################################################

        mel_k3 = F.gelu(self.fc_residual_mel_k3(torch.cat([x_k_3_mel, x[:, 0, :][None, :] ], dim=-1)))
        mel_k5 = F.gelu(self.fc_residual_mel_k5(torch.cat([x_k_5_mel, x[:, 1, :][None, :]], dim=-1)))
        mel_k7 = F.gelu(self.fc_residual_mel_k7(torch.cat([x_k_7_mel, x[:, 2, :][None, :]], dim=-1)))
        mel_k9 = F.gelu(self.fc_residual_mel_k9(torch.cat([x_k_9_mel, x[:, 3, :][None, :]], dim=-1)))

        loudness_k3 = F.gelu(self.fc_residual_loudness_k3(torch.cat([x_k_3_loudness, x[:, 4, :][None, :]], dim=-1)))
        loudness_k5 = F.gelu(self.fc_residual_loudness_k5(torch.cat([x_k_5_loudness, x[:, 5, :][None, :]], dim=-1)))
        loudness_k7 = F.gelu(self.fc_residual_loudness_k7(torch.cat([x_k_7_loudness, x[:, 6, :][None, :]], dim=-1)))
        loudness_k9 = F.gelu(self.fc_residual_loudness_k9(torch.cat([x_k_9_loudness, x[:, 7, :][None, :]], dim=-1)))

        kernels_embs = torch.cat([mel_k3, mel_k5, mel_k7, mel_k9, loudness_k3, loudness_k5, loudness_k7, loudness_k9], dim=0)

        kernels_embs = kernels_embs.transpose(0, 1)

        kernels_embs = kernels_embs.contiguous().view(-1, self.max_node_num * self.out_dim)

        common_embeddings = F.gelu(self.fc_all_nodes_to_classification_embeddings(kernels_embs))

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.gelu(self.fc_embedding_event(common_embeddings))
        scene_embeddings = F.gelu(self.fc_embedding_scene(common_embeddings))

        ISOPls_embeddings = F.gelu(self.fc_embedding_ISOPls(common_embeddings))
        ISOEvs_embeddings = F.gelu(self.fc_embedding_ISOEvs(common_embeddings))

        pleasant_embeddings = F.gelu(self.fc_embedding_pleasant(common_embeddings))
        eventful_embeddings = F.gelu(self.fc_embedding_eventful(common_embeddings))
        chaotic_embeddings = F.gelu(self.fc_embedding_chaotic(common_embeddings))
        vibrant_embeddings = F.gelu(self.fc_embedding_vibrant(common_embeddings))
        uneventful_embeddings = F.gelu(self.fc_embedding_uneventful(common_embeddings))
        calm_embeddings = F.gelu(self.fc_embedding_calm(common_embeddings))
        annoying_embeddings = F.gelu(self.fc_embedding_annoying(common_embeddings))
        monotonous_embeddings = F.gelu(self.fc_embedding_monotonous(common_embeddings))
        # -------------------------------------------------------------------------------------------------------------

        event = self.fc_final_event(event_embeddings)
        scene = self.fc_final_scene(scene_embeddings)

        ISOPls = self.fc_final_ISOPls(ISOPls_embeddings)
        ISOEvs = self.fc_final_ISOEvs(ISOEvs_embeddings)

        pleasant = self.fc_final_pleasant(pleasant_embeddings)
        eventful = self.fc_final_eventful(eventful_embeddings)
        chaotic = self.fc_final_chaotic(chaotic_embeddings)
        vibrant = self.fc_final_vibrant(vibrant_embeddings)
        uneventful = self.fc_final_uneventful(uneventful_embeddings)
        calm = self.fc_final_calm(calm_embeddings)
        annoying = self.fc_final_annoying(annoying_embeddings)
        monotonous = self.fc_final_monotonous(monotonous_embeddings)

        return scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous


