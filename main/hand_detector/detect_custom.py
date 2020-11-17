import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

torch.set_grad_enabled(False)

class YOLO_AKASH(object):

    def __init__(self, 
                weights_path, 
                image_size = 640, 
                conf_thres=0.6, 
                iou_thres=0.3, 
                classes=None, 
                device='', 
                save_img= False,
                augment=False):

        with torch.no_grad():
            self.weights_path = weights_path
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.classes = classes
            self.device = device
            self.save_img = save_img
            self.image_size = image_size
            self.augment = augment

            # Initialize
            set_logging()
            self.device = select_device(self.device)
            if os.path.exists('inference/output/'):
                shutil.rmtree('inference/output/')  # delete output folder
            os.makedirs('inference/output/')  # make new output folder
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model = attempt_load(self.weights_path, map_location=self.device)  # load FP32 model
            self.imgsz = check_img_size(image_size, s=self.model.stride.max())  # check img_size
            if self.half:
                self.model.half()  # to FP16

            #self.dataset = LoadImages('Test_Images', img_size=self.imgsz)

    def initialize_model(self):
        # Load model
        model = attempt_load(self.weights_path, map_location=self.device)  # load FP32 model

        self.imgsz = check_img_size(self.image_size, s=model.stride.max())  # check img_size
        if self.half:
            model.half()  # to FP16
        self.dataset = LoadImages('Test_Images/', img_size=self.imgsz)

        return model

    def detect_images(self):
        with torch.no_grad():
            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            t0 = time.time()
            img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
            o= self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            
            for path, img, im0s, vid_cap in self.dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=self.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', im0s

                    save_path = str(Path('inference/output/') / Path(p).name)
                    txt_path = str(Path('inference/output/') / Path(p).stem) + ('_%g' % self.dataset.frame if self.dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # for *xyxy, conf, cls in reversed(det):
                        #     if self.save_img:  # Add bbox to image
                        #         label = '%s %.2f' % (names[int(cls)], conf)
                        #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print('%sDone. (%.3fFPS)' % (s, 1/(t2 - t1)))

                    # # Save results (image with detections)
                    # if self.save_img:
                    #     if self.dataset.mode == 'images':
                    #         cv2.imwrite(save_path, im0)
                    #     else:
                    #         pass

    def preprocess_image(self, img):
        # loads 1 image from dataset, returns img, original hw, resized hw
        # if img is None:  # not cached
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return img #, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def detect_image(self, im0s):

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        img = self.preprocess_image(im0s)

        print(img.shape)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)

        # Process detections
        boxes = []
        confs = []
        labels = []
        for i, det in enumerate(pred):  # detections per image
            s, im0='', im0s

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                for *xyxy, conf, cls in reversed(det):

                    bbox = [int(xyxy[0]), int(xyxy[1]) ,int(xyxy[2]), int(xyxy[3])]
                    
                    boxes.append(bbox)
                    confs.append(float(conf))
                    labels.append(int(cls))

        t2 = time_synchronized()
        # Print time (inference + NMS)
        print('%sDone. (%.3f FPS)' % (s, 1/(t2 - t1)))

        return boxes, confs, labels

if __name__ == '__main__':
    
    detector = YOLO_AKASH('last.pt', save_img=True)

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        
        boxes, confs, labels = detector.detect_image(frame)

        for b in boxes:
            frame = cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)

        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()