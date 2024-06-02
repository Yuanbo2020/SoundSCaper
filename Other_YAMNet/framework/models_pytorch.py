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



from framework.Yamnet_params import YAMNetParams

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF Slim
    """
    def __init__(self, *args, **kwargs):
        # remove padding argument to avoid conflict
        padding = kwargs.pop("padding", "SAME")
        # initialize nn.Conv2d
        super().__init__(*args, **kwargs)
        self.padding = padding
        assert self.padding == "SAME"
        self.num_kernel_dims = 2
        self.forward_func = lambda input, padding: F.conv2d(
            input, self.weight, self.bias, self.stride,
            padding=padding, dilation=self.dilation, groups=self.groups,
        )

    def tf_SAME_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.kernel_size[dim]

        dilate = self.dilation
        dilate = dilate if isinstance(dilate, int) else dilate[dim]
        stride = self.stride
        stride = stride if isinstance(stride, int) else stride[dim]

        effective_kernel_size = (filter_size - 1) * dilate + 1
        out_size = (input_size + stride - 1) // stride
        total_padding = max(
            0, (out_size - 1) * stride + effective_kernel_size - input_size
        )
        total_odd = int(total_padding % 2 != 0)
        return total_odd, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return self.forward_func(input, padding=0)
        odd_1, padding_1 = self.tf_SAME_padding(input, dim=0)
        odd_2, padding_2 = self.tf_SAME_padding(input, dim=1)
        if odd_1 or odd_2:
            # NOTE: F.pad argument goes from last to first dim
            input = F.pad(input, [0, odd_2, 0, odd_1])

        return self.forward_func(
            input, padding=[ padding_1 // 2, padding_2 // 2 ]
        )


class CONV_BN_RELU(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(
            conv.out_channels, eps=YAMNetParams.BATCHNORM_EPSILON
        )  # NOTE: yamnet uses an eps of 1e-4. This causes a huge difference
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.fused = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False
            )
        )

    def forward(self, x):
        return self.fused(x)

class SeparableConv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.depthwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=input_dim, groups=input_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False,
            ),
        )
        self.pointwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=1,
                padding='SAME', bias=False,
            ),
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class YAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        net_configs = [
            # (layer_function, kernel, stride, num_filters)
            (Conv, [3, 3], 2, 32),
            (SeparableConv, [3, 3], 1, 64),
            (SeparableConv, [3, 3], 2, 128),
            (SeparableConv, [3, 3], 1, 128),
            (SeparableConv, [3, 3], 2, 256),
            (SeparableConv, [3, 3], 1, 256),
            (SeparableConv, [3, 3], 2, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 2, 1024),
            (SeparableConv, [3, 3], 1, 1024)
        ]

        input_dim = 1
        self.layer_names = []
        for (i, (layer_mod, kernel, stride, output_dim)) in enumerate(net_configs):
            name = 'layer{}'.format(i + 1)
            self.add_module(name, layer_mod(kernel, stride, input_dim, output_dim))
            input_dim = output_dim
            self.layer_names.append(name)

        each_emotion_class = 1

        input_dim = 1024
        self.fc_final_event = nn.Linear(input_dim, 15, bias=True)
        self.fc_final_scene = nn.Linear(input_dim, 3, bias=True)

        # MSE
        self.fc_final_ISOPls = nn.Linear(input_dim, 1, bias=True)
        self.fc_final_ISOEvs = nn.Linear(input_dim, 1, bias=True)

        self.fc_final_pleasant = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(input_dim, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(input_dim, each_emotion_class, bias=True)

    def forward(self, x):
        # print(x.size())
        # torch.Size([32, 3001, 64])

        x = x.unsqueeze(1)

        for name in self.layer_names:
            mod = getattr(self, name)
            x = mod(x)
        x = F.adaptive_avg_pool2d(x, 1)

        x = x.reshape(x.shape[0], -1)

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








