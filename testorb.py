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

def measure(orb_extractor, image_loader: ImageLoader) -> None:
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

if __name__ == "__main__":
    orb = cv2.ORB_create()
    dataset_path={"MH01":"/home/nembot/datasets/MH01","MH02":"/home/nembot/datasets/MH02","V101":"/home/nembot/datasets/V101","V201":"/home/nembot/datasets/V201"}

    for name,path in dataset_path.items():
        image_path=path+"/mav0/cam0/data"
        image_loader = ImageLoader(image_path)
        print("Testing ORB Feature Extractor on {} dataset".format(name))
        measure(orb, image_loader)




