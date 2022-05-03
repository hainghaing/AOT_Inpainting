import torch
from torch import nn
from torch.nn import functional as F


def conv_bn(inp, oup, kernel=3, stride=2, padding=1, activation=nn.Sigmoid):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        activation()
    )

class UnetEffb6Generator(nn.Module):
    def __init__(self, input_nc, output_nc, refine):
        super(UnetEffb6Generator, self).__init__()

        basemodel_name = 'tf_efficientnet_b6_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=False)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.encoder = nn.Sequential(
            conv_bn(input_nc, 3, 1, 1, 0),
            Encoder(basemodel)
        )

        # hardcoded input_nc
        self.decoder = nn.Sequential(
            UpConv(200, 256),
            nn.ReLU(True),
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            UpConv(64, 32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, y, **kwargs):
        x = torch.cat([x, y], dim=1)
        x = self.encoder(x)         # array 16
        # x[8] <= 2, 200, 32, 32
        unet_out = self.decoder(x[8], **kwargs)
        return unet_out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for layer_inx, (k, v) in enumerate(self.original_model._modules.items()):
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# AOT-GAN Decoder
class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
