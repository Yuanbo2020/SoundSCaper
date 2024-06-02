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


#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################



class CNN_Transformer(nn.Module):
    def __init__(self, batchnormal=True):

        super(CNN_Transformer, self).__init__()

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

        encoder_layers = 1
        self.mha = Encoder(input_dim=23, n_layers=encoder_layers, output_dim=d_model)

        self.mha_d_model_to_one = nn.Linear(d_model * 128, 128, bias=True)


        each_emotion_class = 1
        d_embeddings = d_model * 128
        self.fc_final_event = nn.Linear(d_embeddings, 15, bias=True)
        # self.event_embedding_layer = nn.Linear(15, 15, bias=True)
        self.fc_final_scene = nn.Linear(d_embeddings, 3, bias=True)

        # MSE
        # self.PAQ_embedding_layer = nn.Linear(8, 8, bias=True)
        self.fc_final_ISOPls = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_ISOEvs = nn.Linear(d_embeddings, each_emotion_class, bias=True)

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

        x = F.relu_(self.bn3(self.conv3(x)))  # (5,5)
        x = F.max_pool2d(x, kernel_size=(3, 2))

        x = F.relu_(self.bn5(self.conv5(x)))  # (7,7)
        x = F.max_pool2d(x, kernel_size=(4, 1))

        x = F.relu_(self.bn7(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=(5, 2))

        x_com, x_scene_self_attns = self.mha(x)
        #################

        x = x_com.view(x_com.size()[0], -1)

        event = self.fc_final_event(x)
        scene = self.fc_final_scene(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        ISOPls = self.fc_final_ISOPls(x)
        ISOEvs = self.fc_final_ISOEvs(x)

        return scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous






