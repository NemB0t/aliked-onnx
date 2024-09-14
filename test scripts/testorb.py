import cv2
from time import time
import glob
import os
import logging
import numpy as np

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

def measure(orb_extractor, image_loader) :
    timings = []
    print("Runing benchmark on {} images".format(len(image_loader)))
    for image in image_loader:
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time()
        kpts, dess = orb.detectAndCompute(image,None)
        end_time = time()
        duration = (end_time - start_time) * 1000  # convert to ms.
        timings.append(duration)

    print(f"mean: {np.mean(timings):.2f}ms")
    print(f"median: {np.median(timings):.2f}ms")
    print(f"min: {np.min(timings):.2f}ms")
    print(f"max: {np.max(timings):.2f}ms")

def keypoints_to_img(keypoints,img_tensor):
    # use if keypoints is a numpy array
    _, _, h, w = img_tensor.shape
    wh = np.array([w - 1, h - 1], dtype=np.float32)

    keypoints = wh * (keypoints + 1) / 2
    return keypoints

# if __name__ == "__main__":
#     orb = cv2.ORB_create()
#     dataset_path={"MH01":"/home/nembot/datasets/MH01","MH02":"/home/nembot/datasets/MH02","V101":"/home/nembot/datasets/V101","V201":"/home/nembot/datasets/V201"}
#
#     for _ in range(5):
#         for name,path in dataset_path.items():
#             image_path=path+"/mav0/cam0/data"
#             image_loader = ImageLoader(image_path)
#             print("Testing ORB Feature Extractor on {} dataset".format(name))
#             measure(orb, image_loader)

image_path="./assets/euroc/1403636620963555584.png"
image_path="/home/nembot/aliked-onnx/assets/euroc/v101/1413393260855760384.png"#Frame 1 V201

#"./assets/euroc/1403636620963555584.png"#"/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"
img=cv2.imread(image_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orb = cv2.ORB_create()
start_time = time()
kpts, dess = orb.detectAndCompute(img,None)
end_time = time()
duration = (end_time - start_time) * 1000
print(f"Time to run orb is : {duration:.2f} ms",duration)
print(f"Number of keypoints: {len(kpts)}")
for kpt in kpts:
  kpt= (int(round(kpt.pt[0])), int(round(kpt.pt[1])))
  cv2.circle(img, kpt, 1, (0, 0, 255), -1, lineType=16)
# print(kpts)
# cv2.imshow("ORB Normal",img)
cv2.imshow("1413393260855760384",img)
cv2.waitKey()