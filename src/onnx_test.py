import onnxruntime
import cv2
import numpy as np

input_image = cv2.imread("test.png")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

ort_session = onnxruntime.InferenceSession("mask.onnx")
ort_input = {ort_session.get_inputs()[0].name: input_image}
mask = ort_session.run(None, ort_input)[0]       # b, h, w, 1

###################### inpainting
# ort_session = onnxruntime.InferenceSession("inpainting.onnx")
ort_session = onnxruntime.InferenceSession("inp512_mob.onnx")

input_image = np.expand_dims(input_image, axis=0)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: input_image, ort_session.get_inputs()[1].name: mask}
ort_outs = ort_session.run(None, ort_inputs)

output = cv2.cvtColor(ort_outs[0], cv2.COLOR_RGB2BGR)
cv2.imwrite("test_resulttttt.jpg", output)