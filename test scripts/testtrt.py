from trt_model import LOGGER_DICT, TRTInference
import cv2
from time import time
from torchvision.transforms import ToTensor
import numpy as np
import glob
import os
import logging
from tqdm import tqdm

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

def measure(engine, image_loader) :
    print("Runing benchmark on {} images".format(len(image_loader)))
    timings = []
    for image in tqdm(image_loader):
        #  image: (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = ToTensor()(image).unsqueeze(0)  # (B, C, H, W)
        image = image_tensor.numpy()

        start_time = time()
        engine.infer(image)
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
# if __name__ == '__main__':
#     trt_model_path ="./converted_trt_models/fp16-aliked-n32-top1k-euroc.onnx"# "converted_trt_models/aliked-n32-top1k-euroc.trt"
#     dataset_path = {"MH01": "/home/nembot/datasets/MH01", "MH02": "/home/nembot/datasets/MH02",
#                     "V101": "/home/nembot/datasets/V101", "V201": "/home/nembot/datasets/V201"}
#     warmup_img_loc = "./assets/euroc/1403636620963555584.png"  # "/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"
#     trt_logger = LOGGER_DICT["verbose"]
#     model_name = "aliked-n32"
#     engine = TRTInference(trt_model_path, model_name, trt_logger)
#
#     #Warming up TRT Engine
#     warmup_image = cv2.imread(warmup_img_loc)
#     warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2RGB)
#     print("Starting warm-up ...")
#     image_tensor = ToTensor()(warmup_image).unsqueeze(0)  # (B, C, H, W)
#     image = image_tensor.numpy()
#     image_w = image
#     num_iterations = 100  # number of iterartions to warmup the model
#     for _ in range(num_iterations):
#         engine.infer(image_w)
#     print("Warm-up done!")
#
#     # for _ in range(5):
#     for name, path in dataset_path.items():
#         print("Testing TensorRT ALIKED Feature Extractor on {} dataset".format(name))
#         image_path = path + "/mav0/cam0/data"
#         image_loader = ImageLoader(image_path)
#         measure(engine, image_loader)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")



model_name="aliked-n32"# modify accordingly aliked-t16,aliked-n16,aliked-n16rot,aliked-n32
imagepath = "../assets/euroc"
single_img_loc="../assets/euroc/1403636620963555584.png"#"/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"


# aliked_service = create_aliked_service(args)
trt_logger = LOGGER_DICT["verbose"]
trt_model_path ="../converted_trt_models/fp16-aliked-n32-top1k-euroc.onnx" #converted_trt_models/aliked-n32-top1k-euroc.trt"
model = TRTInference(trt_model_path, model_name, trt_logger)

#Warming up the model
warmup_image = cv2.imread(single_img_loc)
disp_img=warmup_image
warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2RGB)
# aliked_service.warmup(warmup_image)
image=warmup_image
print("Starting warm-up ...")
image_tensor = ToTensor()(image).unsqueeze(0)  # (B, C, H, W)
image = image_tensor.numpy()
image_w = image
num_iterations=3#number of iterartions to warmup the model
for _ in range(num_iterations):
    model.infer(image_w)
print("Warm-up done!")

# show_memory_gpu_usage()
start_time = time()
pred_trt = model.run(disp_img)
end_time = time()
duration = (end_time - start_time) * 1000  # convert to ms.
print(f"Time to run ALIKED TRT is : {duration:.2f} ms",duration)
# keypoints=keypoints.reshape(-1, 2)
# keypoints=keypoints.cpu().numpy()
# print(pred_trt)
keypoints=pred_trt['keypoints']
print(f"Number of keypoints: {len(keypoints)}")
# show_memory_gpu_usage()
# keypoints=keypoints_to_img(keypoints,image)
for keypoint in keypoints:
    kpt = (int(round(keypoint[0])), int(round(keypoint[1])))
    cv2.circle(disp_img, kpt, 1, (0, 0, 255), -1, lineType=16)
    # print(kpt)
# # print(disp_img.shape)
cv2.imshow("ALIKED TensorRT", disp_img)
cv2.waitKey()