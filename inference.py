import collections
import time
from pathlib import Path
import torch
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadFrame
from utils.general import check_img_size, non_max_suppression, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized


class YoloMask:
    def detect(
               frame=None,
               weights='weights/yolomask_1.pt',
               imgsz=640,
               save_dir=Path(''),  # for saving images
               save_txt=False,  # for auto-labelling
               device='',
               conf_thres=0.45,
               iou_thres=0.5,  # for NMS
               classes=False,
               exist_ok=False
               ):

        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Set Dataloader
        if hasattr(frame, 'img'):
            dataset = LoadFrame(frame.img, img_size=imgsz)
        else:
            raise StopIteration('Frame has no img attribute')

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None
        for _, img, _, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes=classes, agnostic=False)
            time_synchronized()
            
            detections = []
            if hasattr(frame, 'img'):
                print(f'Done. ({time.time() - t0:.3f}s)')
                for _,det in enumerate(pred):
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = xyxy
                        reformat = ((x1.item(), y1.item(), x2.item(), y2.item()),
                            conf.item(), cls.item())
                        detections.append(reformat)
                return detections
        
if __name__ == '__main__':
    class Frame:
        img = cv2.imread('data/images/bibi.png')
    myframe = Frame()
    detections = YoloMask.detect(frame=myframe)
    print(detections)
