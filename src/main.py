import argparse 
import cv2
import numpy as np
import pyrealsense2 as rs

import realsense as rs
import yolo 

from utils.plots import plot_one_box

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../customize_yolo/weights/default/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='don`t trace model')
    cfg = parser.parse_args()
    print(cfg)

    D435 = rs.RealsenseCamera()
    vision = yolo.Yolo(cfg)

    # Intrinsic camera parameters
    ppx, ppy = D435.get_optical_center()
    fx, fy = D435.get_focal_center()

    while(True):
        flag, bgr_frame, depth_frame = D435.get_frame_stream()

        bgr_image = np.asanyarray(bgr_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        pred = vision.detect(bgr_frame)

        if len(pred):
            classes, boxes, centers, confidences = vision.process_prediction(pred)

            for cls, box, center, conf in zip(classes, boxes, centers, confidences):
                cx, cy = center[0], center[1]
                depth_mm = depth_frame.get_distance(int(cx), int(cy)) * 100

                # calculate real world coordinates
                Z = depth_mm
                X = Z * (cx - ppx) / fx
                Y = Z * (cy - ppy) / fy

                coordinates_text = "("   + str(round(X,2)) + \
                                    ", " + str(round(Y,2)) + \
                                    ", " + str(round(Z,2)) + ")"

                cv2.putText(bgr_image, coordinates_text, (int(cx) - 160, int(cy)),
                            0, 0.8, (255, 255, 255), 2)

                c = int(cls)  # integer class
                label = f'{conf:.2f}'
                plot_one_box(box, bgr_image, label=label, color=int(cls), line_thickness=2)

        # Stream results
        cv2.imshow("Recognition result", bgr_image)
        cv2.imshow("Recognition result depth", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
