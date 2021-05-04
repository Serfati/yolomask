import time
import torch
import lib.config as globals

from models.experimental import attempt_load
from utils.datasets import LoadFrame
from utils.general import check_img_size, non_max_suppression, set_logging, scale_coords
from utils.torch_utils import select_device


class YoloPerson:
    def __init__(self,
                 weights='yolomask/weights/yolov5s.pt',
                 imgsz=640,
                 device='',
                 conf_thres=0.45,
                 iou_thres=0.5,
                 classes=None):
        if classes is None: classes = [0]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # load FP32 model
        self.model = attempt_load(weights, map_location=self.device)
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

    def detect(self, frame=None):
        # Set Dataloader
        if hasattr(frame, 'img'):
            dataset = LoadFrame(frame.img, img_size=self.imgsz)
        else:
            raise StopIteration('Frame has no img attribute')

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz),
                          device=self.device)  # init img
        # run once
        _ = self.model(
            img.half() if self.half else img) if self.device.type != 'cpu' else None
        for _, img, img0, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)

            detections = []
            for _, det in enumerate(pred):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    reformat = ((x1.item(), y1.item(), x2.item(), y2.item()),
                                conf.item())
                    detections.append(reformat)
            frame.persons = detections
            return detections
