import cv2
from torchvision.transforms import ToTensor
import onnxruntime
import onnx
from tqdm import tqdm
from time import time
import torch
import numpy as np
import glob
import os
import logging

class ImageLoader(object):
    def __init__(self, filepath: str):
        self.images = (
            glob.glob(os.path.join(filepath, "*.png"))
            + glob.glob(os.path.join(filepath, "*.jpg"))
            + glob.glob(os.path.join(filepath, "*.ppm"))
        )
        self.images.sort()
        self.num_images = len(self.images)
        logging.info("Loading %s images", {self.num_images})
        self.mode = "images"

    def __getitem__(self, item):
        filename = self.images[item]
        img = cv2.imread(filename)
        return img

    def __len__(self):
        return self.num_images

def measure(ort_session, image_loader) :
    print("Runing benchmark on {} images".format(len(image_loader)))
    timings = []
    for image in tqdm(image_loader):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = (
            ToTensor()(image).to("cuda").unsqueeze(0)
        )
        input_data = [image_tensor]
        defined_inputs = ort_session.get_inputs()
        ort_inputs = {
            defined_inputs[index].name: to_numpy(input_data[index])
            for index in range(len(defined_inputs))
        }

        start_time = time()

        pred_onnx = ort_session.run(
            [
                "keypoints",
                "descriptors",
                "scores",
            ],
            ort_inputs,
        )

        end_time = time()
        duration = (end_time - start_time) * 1000  # convert to ms.
        timings.append(duration)

    print(f"mean: {np.mean(timings):.2f}ms")
    print(f"median: {np.median(timings):.2f}ms")
    print(f"min: {np.min(timings):.2f}ms")
    print(f"max: {np.max(timings):.2f}ms")

def to_numpy(tensor: torch.Tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

def keypoints_to_img(keypoints,img_tensor):
    # use if keypoints is a numpy array
    _, _, h, w = img_tensor.shape
    wh = np.array([w - 1, h - 1], dtype=np.float32)

    keypoints = wh * (keypoints + 1) / 2
    return keypoints

# if __name__ == '__main__':
#     onnx_model_path = "./converted_onnx_models/aliked-n32-top1k-euroc.onnx"
#     dataset_path = {"MH01": "/home/nembot/datasets/MH01", "MH02": "/home/nembot/datasets/MH02",
#                     "V101": "/home/nembot/datasets/V101", "V201": "/home/nembot/datasets/V201"}
#
#     for _ in range(5):
#         for name, path in dataset_path.items():
#             print("Testing Onnx ALIKED Feature Extractor on {} dataset".format(name))
#             image_path = path + "/mav0/cam0/data"
#             image_loader = ImageLoader(image_path)
#             ort_session = onnxruntime.InferenceSession(onnx_model_path)
#             measure(ort_session,image_loader)
#         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# onnx_model = onnx.load("./converted_model/aliked-n32-top1k-euroc.onnx")
# onnx.checker.check_model(onnx_model)
# print(onnx_model)


onnx_model_path = "../converted_onnx_models/aliked-n32-top1k-euroc.onnx"
image_path="../assets/euroc/1403636620963555584.png"
#"/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"
img=cv2.imread(image_path)
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
input_shape = ort_session.get_inputs()[0].shape
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
    print(f"Time to run ALIKED onnx is : {duration:.2f} ms",duration)

# print(image_tensor.shape)
kpts=pred_onnx[0]
# print(kpts.device)
# print(kpts)
kpts=keypoints_to_img(kpts,image_tensor)
print(f"Number of keypoints: {len(kpts)}")
for kpt in kpts:
  kpt= (int(round(kpt[0])), int(round(kpt[1])))
  cv2.circle(img, kpt, 1, (0, 0, 255), -1, lineType=16)
  # print(kpt)
cv2.imshow("ALIKED ONNX",img)
cv2.waitKey()
# print(pred_onnx[2])




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
