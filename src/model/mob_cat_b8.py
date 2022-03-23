import logging

import torch
import torch.nn as nn
from model.aotgan import AOTBlock

def conv_bn(inp, oup, kernel=3, stride=2, padding=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        activation()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
class UnetMobileGenerator(nn.Module):
    def __init__(self, args, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=True):
        super(UnetMobileGenerator, self).__init__()

        self.backbone = MobileNetV2(n_channels=input_nc, n_classes=3)

        # AOT-Block
        self.aotblock = nn.Sequential(*[AOTBlock(1280, args.rates) for _ in range(args.block_num)])

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 3, padding=1, stride=2)
        # self.dconv1 = nn.ConvTranspose2d(256, 96, 3, padding=1, stride=2)
        self.invres1 = InvertedResidual(  192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(  96, 32, 3, padding=1, stride=2)
        self.invres2 = InvertedResidual(   64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(  32, 24, 3, padding=1, stride=2)
        self.invres3 = InvertedResidual(   48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(  24, 16, 3, padding=1, stride=2)
        self.invres4 = InvertedResidual(   32, 16, 1, 6)
        ###!###
        self.dconv5 = nn.ConvTranspose2d(  16, 8, 4, padding=1, stride=2)
        # self.invres5 = InvertedResidual(   16, 8, 1, 6)  # inp, oup, stride, expand_ratio

        # self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_last = nn.Sequential(
                            nn.Conv2d(8, output_nc, 1),
                            nn.Tanh()
        )

        # self.conv_score = nn.Conv2d(n_channels, n_classes, 1)

    def forward(self, x, y):  
        x = torch.cat([x, y], dim=1)
        
        for n in range(0, 2):
            x = self.backbone.features[n](x)
            # print(n, x.size())

        x1 = x
        # print(x1.shape, 'x1')
        logging.debug((x1.shape, 'x1'))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
            # print(n, x.size())

        x2 = x
        # print(x2.shape, 'x2')
        logging.debug((x2.shape, 'x2'))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
            # print(n, x.size())

        x3 = x
        # print(x3.shape, 'x3')
        logging.debug((x3.shape, 'x3'))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
            # print(n, x.size())

        x4 = x
        # print(x4.shape, 'x4')
        logging.debug((x4.shape, 'x4'))

        for n in range(14, 19):
            x = self.backbone.features[n](x)
            print(n, x.size())

        x5 = x
        # print(x5.shape, 'x5')
        logging.debug((x5.shape, 'x5'))

        ## AOT Block 추가
        x = self.aotblock(x)

        logging.debug((x4.shape, self.dconv1(x).shape, 'up1'))
        up1 = torch.cat([
            x4,
            self.dconv1(x, output_size=x4.size())
        ], dim=1)
        up1 = self.invres1(up1)
        # print(up1.shape, 'up1')
        logging.debug((up1.shape, 'up1'))
        
        up2 = torch.cat([
            x3,
            self.dconv2(up1, output_size=x3.size())
        ], dim=1)
        up2 = self.invres2(up2)
        # print(up2.shape, 'up2')
        logging.debug((up2.shape, 'up2'))

        up3 = torch.cat([
            x2,
            self.dconv3(up2, output_size=x2.size())
        ], dim=1)
        up3 = self.invres3(up3)
        # print(up3.shape, 'up3')
        logging.debug((up3.shape, 'up3'))

        up4 = torch.cat([
            x1,
            self.dconv4(up3, output_size=x1.size())
        ], dim=1)
        up4 = self.invres4(up4)
        # print(up4.shape, 'up4')
        logging.debug((up4.shape, 'up4'))
        
        up5 = self.dconv5(up4)
        # print(up5.shape, 'up5')
        logging.debug((up5.shape, 'up5'))

        x = self.conv_last(up5)
        # print(x.shape, 'conv_last')
        logging.debug((x.shape, 'conv_last'))

        return x





class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        # print(x.shape)
        # print(self.conv(x).shape)
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_channels=5, n_classes=3, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        # last_channel = 256
        last_channel = 1280
        
        ''' t : expansion factor
            c : output channel의 수
            n : 반복 횟수
            s : stride          '''
        interverted_residual_setting = [
            # t, c, n, s 
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(n_channels, input_channel)]
        
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
