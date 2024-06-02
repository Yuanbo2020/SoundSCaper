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



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self):

        super(MobileNetV2, self).__init__()


        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        each_emotion_class = 1

        d_embeddings = 1280
        self.fc_final_event = nn.Linear(d_embeddings, 15, bias=True)
        self.fc_final_scene = nn.Linear(d_embeddings, 3, bias=True)

        # MSE
        self.fc_final_ISOPls = nn.Linear(d_embeddings, 1, bias=True)
        self.fc_final_ISOEvs = nn.Linear(d_embeddings, 1, bias=True)

        self.fc_final_pleasant = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(d_embeddings, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(d_embeddings, each_emotion_class, bias=True)


    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = input.unsqueeze(1)  # torch.Size([64, 1, 480, 64])


        x = self.features(x)

        x = torch.mean(x, dim=3)

        x = torch.mean(x, dim=2)

        event = self.fc_final_event(x)
        scene = self.fc_final_scene(x)

        ISOPls = self.fc_final_ISOPls(x)
        ISOEvs = self.fc_final_ISOEvs(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        return scene, event, ISOPls, ISOEvs, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous






