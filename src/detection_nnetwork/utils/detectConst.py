NAME = 'exp'
PROJECT = 'runs/detectc'
EXIST_OK = True

WEIGHTS = '../customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
NO_TRACE =  True 
DEVICE = '0' # CUDA DEVICE
IMG_SIZE = int(800)

AUGMENT = True
CONF_THRES = float(0.75)
IOU_THRES = float(0.45)

CLASSES = None

AGNOSTIC_NMS = True

NOSAVE = True
SAVE_CONF = True
VIEW_IMAGE = False 
SAVE_TXT = False

UPDATE = True