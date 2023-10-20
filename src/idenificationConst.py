import cv2 as cv

TYPE = 1
# TYPE 0 := W/O DATA STREAM
# TYPE 1 := HWIL TEST

PIPELINE, CAP_OPT, SUB_ADDR, LOG_FILE, DNN_WEIGHTS = None, None, None, None, None

if( TYPE == 1 ):
    PIPELINE    =  r'udpsrc port=1234 ! video/mpegts ! tsdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false drop=true max-buffers=1'
    CAP_OPT     =  cv.CAP_GSTREAMER
    SUB_ADDR    =  r"tcp://127.0.0.1:5555"
    LOG_FILE    =  r'./data/telemetry.csv'
    DNN_WEIGHTS =  r'./detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'