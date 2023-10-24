NAME = 'exp'
PROJECT = 'runs/detectc'
EXIST_OK = False

WEIGHTS = '../customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
NO_TRACE =  False 
DEVICE = '0' # CUDA DEVICE
IMG_SIZE = int(800)

AUGMENT = False
CONF_THRES = float(0.75)
IOU_THRES = float(0.45)

CLASSES = None

AGNOSTIC_NMS = False

NOSAVE = True
SAVE_CONF = False
VIEW_IMAGE = False 
SAVE_TXT = False

UPDATE = False