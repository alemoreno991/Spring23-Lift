import cv2 as cv

class ids_opt:

    pipeline, cap_opt, ds_sub_address_string, logger_file_string, dnn_weights, classification_type = None, None, None, None, None, None

    def __init__(self, type ):
        if(   type == 0 ):  # SINGLE IMAGE INPUT
            self.dnn_weights            =  r'./identification_stack/marker_identification/detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
            self.classification_type    =  r'utlift'

        elif( type == 1 ):  # HWIL STREAMED INPUTS
            self.pipeline               =  r'udpsrc port=1234 ! video/mpegts ! tsdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false drop=true max-buffers=1'
            self.cap_opt                =  cv.CAP_GSTREAMER
            self.ds_sub_address_string  =  r"tcp://127.0.0.1:5555"
            self.logger_file_string     =  r'./data/telemetry.csv'
            self.dnn_weights            =  r'./identification_stack/marker_identification/detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
            self.classification_type    =  r'utlift'


        elif( type == 2 ):  # INTEL REALSENSE W/O DATA STREAM
            self.cap_opt                =  r'IRSD455'
            self.dnn_weights            =  r'./identification_stack/marker_identification/detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
            self.classification_type    =  r'utlift'

    ###################################################################################################################################################################################################################################################################################

        if(   type == r'A' ):  # SINGLE IMAGE INPUT
            # self.dnn_weights            =  r'./identification_stack/marker_identification/detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
            self.classification_type    =  r'aruco'

        elif( type == r'B' ):  # HWIL STREAMED INPUTS
            self.pipeline               =  r'udpsrc port=1234 ! video/mpegts ! tsdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false drop=true max-buffers=1'
            self.cap_opt                =  cv.CAP_GSTREAMER
            self.ds_sub_address_string  =  r"tcp://127.0.0.1:5555"
            self.logger_file_string     =  r'./data/telemetry.csv'
            # self.dnn_weights            =  r'./identification_stack/marker_identification/detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
            self.classification_type    =  r'aruco'

        elif( type == r'C' ):  # INTEL REALSENSE W/O DATA STREAM
            self.cap_opt                =  r'IRSD455'
            # self.dnn_weights            =  r'./identification_stack/marker_identification/detection_nnetwork/customize_yolo/runs/train/yolov7-tiny-custom/weights/best.pt'
            self.classification_type    =  r'aruco'


        else:
            print( "[WARNING]: INVALID OR ALL-NONE TYPE SPECIFIED " )
            pass
        

