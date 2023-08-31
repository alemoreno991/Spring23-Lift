import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np



if __name__ == "__main__":
    import utils.detectConst as dctc
    from src.yolov7.models.experimental import attempt_load
    from src.yolov7.utils.datasets import letterbox
    from src.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from src.yolov7.utils.plots import plot_one_box
    from src.yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel
else:
    from .utils import detectConst as dctc 
    from .src.yolov7.models.experimental import attempt_load
    from .src.yolov7.utils.datasets import letterbox
    from .src.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from .src.yolov7.utils.plots import plot_one_box
    from .src.yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel


class detectionNetwork:

    # CONSTRUCTOR (LOAD MODEL)
    def __init__(self, weights = dctc.WEIGHTS):
        # logging
        set_logging()

        # Initialize
        self.trace, self.device, self.view_img, self.imgsz, self.save_img, self.save_txt = not dctc.NO_TRACE, select_device(dctc.DEVICE), dctc.VIEW_IMAGE, dctc.IMG_SIZE, not dctc.NOSAVE, dctc.SAVE_TXT
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.save_dir = Path(increment_path(Path(dctc.PROJECT) / dctc.NAME, exist_ok=dctc.EXIST_OK))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        if self.trace:
            self.model = TracedModel( self.model, self.device, self.imgsz )
        if self.half:
            self.model.half()  # to FP16

        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]



    def imageDetection( self, im0, save_name = 'one'):
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()

        # Convert Image
        img = letterbox(im0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=dctc.AUGMENT)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=dctc.AUGMENT)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, dctc.CONF_THRES, dctc.IOU_THRES, classes=dctc.CLASSES, agnostic=dctc.AGNOSTIC_NMS)
        t3 = time_synchronized()

        # Process detections
        xyxy_list = []
        for i, det in enumerate(pred):  # detections per image
            s = ''
            save_path = str(self.save_dir / (save_name + '.jpg') )  # img.jpg
            txt_path = str(self.save_dir / 'labels' / save_name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_list.append( xyxy )

                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if dctc.SAVE_CONF else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.view_img:  # Add bbox to image
                        label = f'UTLift Crate ({conf:.2f})'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if self.view_img:
                cv2.imshow( save_name, im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if self.save_img:
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")

        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            #print(f"Results saved to {self.save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

        return xyxy_list


# if __name__ == '__main__':
#     with torch.no_grad():
#         if dctc.UPDATE:  # update all models (to fix SourceChangeWarning)
#             for dctc.WEIGHTS in ['yolov7.pt']:
#                 detectc()
#                 strip_optimizer(dctc.WEIGHTS)
#         else:
#             detectc()
