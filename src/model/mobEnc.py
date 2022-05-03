import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class UnetMobileEncGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=True):
        super(UnetMobileEncGenerator, self).__init__()

        self.backbone = MobileNetV2(n_channels=input_nc, n_classes=3)

        ## decoder
        self.decoder = nn.Sequential(
            UpConv(1280, 640),
            nn.ReLU(True),
            UpConv(640, 320),
            nn.ReLU(True),
            UpConv(320, 160),
            nn.ReLU(True),
            UpConv(160, 80),
            nn.ReLU(True),
            UpConv(80, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, y):  
        x = torch.cat([x, y], dim=1)
        
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        # print(x.shape, 'x1')

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        # print(x.shape, 'x2')

        for n in range(4, 7):
            x = self.backbone.features[n](x)
       # print(x.shape, 'x3')

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        # print(x.shape, 'x4')

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        # print(x.shape, 'x5')       # 2, 1280, 16, 16

        ##################################################################### [END] Encoder

        x = self.decoder(x)     ## 2, 3, 512, 512
        x = torch.tanh(x)

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
