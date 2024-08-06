from trt_model import LOGGER_DICT, TRTInference
import cv2
from gpu_utils import show_memory_gpu_usage
from time import time
from torchvision.transforms import ToTensor
import numpy as np


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
def main():
    trt_model_path ="converted_trt_models/aliked-n32-top1k-euroc.trt"
    model_name="aliked-n32"# modify accordingly aliked-t16,aliked-n16,aliked-n16rot,aliked-n32
    imagepath = "./assets/euroc"
    single_img_loc="./assets/euroc/1403636620963555584.png"#"/home/nembot/Datasets/EuRoc/MH01/mav0/cam0/data/1403636642563555584.png"


    # aliked_service = create_aliked_service(args)
    trt_logger = LOGGER_DICT["verbose"]
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
    # show_memory_gpu_usage()
    # keypoints=keypoints_to_img(keypoints,image)
    for keypoint in keypoints:
        kpt = (int(round(keypoint[0])), int(round(keypoint[1])))
        cv2.circle(disp_img, kpt, 1, (0, 0, 255), -1, lineType=16)
        # print(kpt)
    # # print(disp_img.shape)
    cv2.imshow("", disp_img)
    cv2.waitKey()


if __name__ == "__main__":
    main()