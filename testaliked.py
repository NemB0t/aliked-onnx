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

image_path="./assets/euroc/1403636620963555584.png"#"/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"
img=cv2.imread(image_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model= ALIKED(model_name="aliked-n32",top_k=1000)
start_time = time()
pred_aliked = model.run(img)
end_time = time()
duration = (end_time - start_time) * 1000
print(f"Time to run ALIKED pytorch is : {duration:.2f} ms",duration)
kpts = pred_aliked['keypoints']
for kpt in kpts:
  kpt= (int(round(kpt[0])), int(round(kpt[1])))
  cv2.circle(img, kpt, 1, (0, 0, 255), -1, lineType=16)
  # print(to_numpy(kpt))
cv2.imshow("",img)
cv2.waitKey()
