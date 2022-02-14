import importlib
from matplotlib.pyplot import axis
import numpy as np
from PIL import Image
import onnxruntime
import cv2

import torch
from torchvision.transforms import ToTensor

from utils.option import args 


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image
    # return Image.fromarray(image)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main_worker(args, use_gpu=True): 

    device = torch.device('cpu')
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location=device))
    model.eval()
    
    onnx_input = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    onnx_input = cv2.cvtColor(onnx_input, cv2.COLOR_BGR2RGB)
    onnx_mask = cv2.imread("test_mask.jpg", cv2.IMREAD_GRAYSCALE)

    onnx_input = np.expand_dims(onnx_input, axis=0)
    onnx_mask = np.expand_dims(onnx_mask, axis=-1)
    onnx_mask = np.expand_dims(onnx_mask, axis=0)

    # iteration through datasets
    image = ToTensor()(Image.open("test.jpg").convert('RGB'))
    image = (image * 2.0 - 1.0).unsqueeze(0)
    print(image.size())
    mask = ToTensor()(Image.open("test_mask.jpg").convert('L'))
    mask = mask.unsqueeze(0)
    print(mask.size())
    image_masked = image * (1 - mask.float()) + mask
    print(image_masked.size())
    
    with torch.no_grad():
        pred_img = model(image_masked, mask)

    # print(pred_img.size())
    # exit()
    output = postprocess(pred_img[0])
    # output.save("result.png", 'png')
    # exit()

    # #######################################################################################
    ort_session = onnxruntime.InferenceSession("sample.onnx")

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input, ort_session.get_inputs()[1].name: onnx_mask}
    ort_outs = ort_session.run(None, ort_inputs)

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    ort_outs[0] = cv2.cvtColor(ort_outs[0], cv2.COLOR_RGB2BGR)

    cv2.imwrite("aa.jpg",output )
    cv2.imwrite("bb.jpg", ort_outs[0])


if __name__ == '__main__':
    main_worker(args)
