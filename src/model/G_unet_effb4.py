import torch
from torch import nn
from torch.nn import functional as F


def conv_bn(inp, oup, kernel=3, stride=2, padding=1, activation=nn.Sigmoid):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        activation()
    )

class UnetEffb4Generator(nn.Module):
    def __init__(self, input_nc, output_nc, refine):
        super(UnetEffb4Generator, self).__init__()

        basemodel_name = 'tf_efficientnet_b4_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=False)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.encoder = nn.Sequential(
            conv_bn(input_nc, 3, 1, 1, 0),
            Encoder(basemodel)
        )

        self.decoder = DecoderBN(output_nc, refine)

    def forward(self, x, y, **kwargs):
        x = torch.cat([x, y], dim=1)
        unet_out = self.decoder(self.encoder(x), **kwargs)
        return unet_out

# This efficient-Unet is adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class DecoderBN(nn.Module):
    def __init__(self, num_classes, refine, num_features=256, bottleneck_features=160):
        super(DecoderBN, self).__init__()
        n_features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, n_features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=368, output_features=n_features // 2)
        self.up2 = UpSampleBN(skip_input=184, output_features=n_features // 4)
        self.up3 = UpSampleBN(skip_input=96, output_features=n_features // 8)
        self.up4 = UpSampleBN(skip_input=35, output_features=n_features // 16)
        self.conv3 = nn.Conv2d(n_features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # if this model is used for transfer learning , remove tanh
        if refine:
            self.act_out = nn.Sequential(nn.BatchNorm2d(num_classes),
                                        nn.LeakyReLU()
            )
        else:
            self.act_out = nn.Tanh()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[7], features[8]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, features[0])
        out = self.conv3(x_d4)

        return self.act_out(out)

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
