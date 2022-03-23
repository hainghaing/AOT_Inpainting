import importlib
import numpy as np
from PIL import Image
from glob import glob

import torch

from utils.option import args 


class Exporter(torch.nn.Module):
    def __init__(self, model):
        super(Exporter, self).__init__()
        self.trainmodel = model

    def forward(self, x, y):
        #Preprocess
        
        x = x[:, :, :, :3]
        x = x.to(torch.float32)
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = x * 2.0 - 1.0

        y = y.to(torch.float32)
        y = y / 255.0
        y = y.permute(0, 3, 1, 2)

        x = x * (1 - y) + y

        output = self.trainmodel(x, y)[0]

        output = torch.clamp(output, -1., 1.)
        output = (output + 1) / 2.0 * 255.0
        output = output.permute(1, 2, 0).to(torch.uint8)

        return output


def main_worker(args, use_gpu=False): 

    device = torch.device('cpu')
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.UnetEffb4Generator(4, 3, False).cuda()
    # model = net.UnetMobileGenerator(args, 4, 3).cuda()
    # model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location=device))

    exporterModel = Exporter(model)

    exporterModel.to(device="cpu")
    exporterModel.eval()

    input1 = torch.zeros((1, 512, 512, 3), dtype=torch.uint8)
    input2 = torch.zeros((1, 512, 512, 1), dtype=torch.uint8)
    dummy_input = (input1, input2)

    with torch.no_grad():
        torch.onnx.export(
            exporterModel, dummy_input, "effb4_cat.onnx",
            verbose = True,
            do_constant_folding = True,
            opset_version = 12,
            input_names = ["input", "mask"],
            output_names = ["output"],
            dynamic_axes={
                'input' : {1 : 'width', 2 : 'height', 3 : 'channels'},
                'mask' : {1 : 'width', 2 : 'height', 3 : 'channels'},
                'output' : {0 : 'width', 1 : 'height'}
                }
        )


if __name__ == '__main__':
    main_worker(args)
