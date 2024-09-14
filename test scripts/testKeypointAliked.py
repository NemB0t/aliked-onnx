import cv2
from tqdm import tqdm
import numpy as np
import glob
import os
import logging
from nets.aliked import ALIKED
import warnings
warnings.filterwarnings("ignore")

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

def measure(aliked_model, image_loader) :
    print("Runing benchmark on {} images".format(len(image_loader)))
    keypoint_count = []
    for image in tqdm(image_loader):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = aliked_model.run(image)['keypoints']
        keypoint_count.append(len(keypoints))
        # print(len(keypoints))


    print(f"mean: {np.mean(keypoint_count):.2f} keypoints")
    print(f"median: {np.median(keypoint_count):.2f} keypoints")
    print(f"min: {np.min(keypoint_count):.2f} keypoints")
    print(f"max: {np.max(keypoint_count):.2f} keypoints")
def keypoints_to_img(keypoints,img_tensor):
    # use if keypoints is a numpy array
    _, _, h, w = img_tensor.shape
    wh = np.array([w - 1, h - 1], dtype=np.float32)

    keypoints = wh * (keypoints + 1) / 2
    return keypoints

# if __name__ == "__main__":
#     dataset_path = {"MH01": "/home/nembot/datasets/MH01", "MH02": "/home/nembot/datasets/MH02",
#                     "V101": "/home/nembot/datasets/V101", "V201": "/home/nembot/datasets/V201"}
#
#     for _ in range(5):
#         for name, path in dataset_path.items():
#             print("Testing pytorch ALIKED Feature Extractor on {} dataset".format(name))
#             image_path = path + "/mav0/cam0/data"
#             image_loader = ImageLoader(image_path)
#             model = ALIKED(model_name="aliked-n32", top_k=-1,scores_th=0.05)
#             measure(model,image_loader)
#         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

image_path = "/home/nembot/aliked-onnx/assets/euroc/v101/low_brightness.png" #low brightness
# image_path="/home/nembot/aliked-onnx/assets/euroc/v101/1413393261355760384.png" # frame2 v201

#"./assets/euroc/1403636620963555584.png"#"/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"
img=cv2.imread(image_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model= ALIKED(model_name="aliked-n32",scores_th=0.035)#,top_k=1000)#top_k=1000
# start_time = time()
pred_aliked = model.run(img)
# end_time = time()
# duration = (end_time - start_time) * 1000
# print(f"Time to run ALIKED pytorch is : {duration:.2f} ms",duration)
kpts = pred_aliked['keypoints']
for kpt in kpts:
  kpt= (int(round(kpt[0])), int(round(kpt[1])))
  cv2.circle(img, kpt, 1, (0, 0, 255), -1, lineType=16)
# print(len(kpts))
# cv2.imshow("ALIKED Low Brightness",img)
cv2.imshow("1413393261355760384",img)
cv2.waitKey()