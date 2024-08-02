import onnx
import argparse
import cv2
from torchvision.transforms import ToTensor
import onnxruntime
from tqdm import tqdm
from time import time
import torch
import numpy as np

from nets.aliked import ALIKED

def to_numpy(tensor: torch.Tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

def keypoints_to_img(keypoints,img_tensor):
    # use if keypoints is a tensor.tensor
    # _, _, h, w = img_tensor.shape
    # wh = torch.tensor([w - 1, h - 1], device=keypoints.device)
    # keypoints = wh * (keypoints + 1) / 2

   # use if keypoints is a numpy array
    _, _, h, w = img_tensor.shape
    wh = np.array([w - 1, h - 1], dtype=np.float32)

    keypoints = wh * (keypoints + 1) / 2
    return keypoints

onnx_model_path = "./converted_onnx_models/aliked-n32-top1k-euroc.onnx"
# onnx_model = onnx.load("./converted_model/aliked-n32-top1k-euroc.onnx")
# onnx.checker.check_model(onnx_model)
# print(onnx_model)

img=cv2.imread("/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_tensor = (
        ToTensor()(img).to("cuda").unsqueeze(0)
    )  # (B, C, H, W)
ort_session = onnxruntime.InferenceSession(onnx_model_path)
defined_inputs = ort_session.get_inputs()
input_data = [image_tensor]
ort_inputs = {
        defined_inputs[index].name: to_numpy(input_data[index])
        for index in range(len(defined_inputs))
    }
for _ in tqdm(range(1), desc="ONNX timing"):
    start_time = time()
    pred_onnx = ort_session.run(
        [
            "keypoints",
            "descriptors",
            "scores",
            # "score_dispersity",
            # "score_map",
        ],
        ort_inputs,
    )
    end_time = time()
    duration = (end_time - start_time) * 1000
    print("Time to run ALIKED onnx is :"+str(duration))

# print(image_tensor.shape)
kpts=pred_onnx[0]
# print(kpts.device)
kpts=keypoints_to_img(kpts,image_tensor)
for kpt in kpts:
  kpt= (int(round(kpt[0])), int(round(kpt[1])))
  cv2.circle(img, kpt, 1, (0, 0, 255), -1, lineType=16)
  # print(kpt)
cv2.imshow("",img)
cv2.waitKey()
print(pred_onnx[2])




# model= ALIKED()
# start_time = time()
# pred_aliked = model.run(img)
# end_time = time()
# duration = (end_time - start_time) * 1000
# print("Time to run ALIKED Pytorch is :"+str(duration))
# kpts=pred_aliked[0]
# for kpt in kpts:
  # kpt= (int(round(kpt[0])), int(round(kpt[1])))
  # cv2.circle(img, kpt, 1, (0, 0, 255), -1, lineType=16)
  # print(to_numpy(kpt))
# cv2.imshow("",img)
# cv2.waitKey()
