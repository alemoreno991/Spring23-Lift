import argparse
import csv 

import cv2 as cv

# DETECTION NNETWORK
import detection_nnetwork.detectc as dtctnn

# CLASSIFICATION FILTER
import classification_filter.decodeImage as cfilter

# ESTIMATION 
import marker_localization.process_measurements as mrklcl

# SERIAL CONNECTION ICP PACKAGES
import zmq
import struct

# SERIAL CONNECTION OPTIONS
IPC_TOPIC = "veronte_tele"
PYSTRUCT_PACKETFORMAT = "20f"
TEST_INIT = False
UI16_PACKET_SIZE = 0
UI32_PACKET_SIZE = 0 
F32_PACKET_SIZE = 20


# def parseTelemetryStreamPackets(inI):
#     outI = inI
#     return outI

def runIdentifiaction_hwil():

    # PREPARE CAMERA STREAM CONNECTION
    print('[INFO]: ATTEMPTING TO OPEN CAPTURE OBJECT')
    cap_obj = cv.VideoCapture( 'udpsrc port=1234 ! video/mpegts ! tsdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false drop=true max-buffers=1' , cv.CAP_GSTREAMER )
    if not cap_obj.isOpened():
        print('[ERROR]: CAPTURE OBJECT DID NOT OPEN ... EXITING')
        exit(0)
    else:
        print('[INFO]: CAPTURE OBJECT IS OPENNED. CONTINUING ...')

    # PREPARE TELEMETRY SERIAL CONNECTION
    struct_format = PYSTRUCT_PACKETFORMAT
    ipc_topic_enc = IPC_TOPIC.encode('ascii')
    print("[INFO] READING IPC MESSAGES ON ZMQ TOPIC: {}".format(ipc_topic_enc))

    # OPEN CONNECTIONS AND START READING
    with zmq.Context() as context, open('./data/telemetry.csv','w') as fout:
        # SETUP SERIAL CONNECTION
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, ipc_topic_enc)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect("tcp://127.0.0.1:5555")

        # PREPARE CSV WRITER 
        csv_out = csv.writer(fout)

        # ITERATOR FOR SAVING TASKS
        i = 0
        
        # PRIMARY LOOP
        print("[INFO]: STARTING PRIMARY LOOP")
        try:
            while True:
                
                # GRAB IMAGE AND TELEMETRY DATA
                _ , data_buffer = socket.recv().split(b' ', 1)
                _, imgi = cap_obj.read()
                data = struct.unpack(struct_format, data_buffer)

                # DETECT MARKERS IN IMAGE
                bbcrnrs_list = detectNN.imageDetection(imgi)

                # CLASSIFY MARKERS IN IMAGE
                for bbcrnr in bbcrnrs_list:
                    
                    read_code = []
                    try:
                        read_code = cfilter.decodeImageSection(imgi,bbcrnr,0,True)
                    except:
                        # TODO: FIND POSSIBILITY FOR INDEXING ERROR IN CLASSIFICATION FILTER
                        print("[ERROR]: AN UNEXPECTED ERROR OCCURED DURING IMAGE DECODING. SKIPPING FRAME ...")
                        continue
                    
                    if len(read_code) > 0:
                        # SAVE TELEMETRYi and IMAGEi FOR ESTIMATION TESTS
                        csv_out.writerow( data )
                        cv.imwrite("./data/images/img_{}.jpg".format(i),imgi)
                        i += 1

                if i >= 100:
                    break
                        
        except KeyboardInterrupt:
            socket.close()

        except Exception as error:
            print("[ERROR]: {}".format(error))
            socket.close()

    cap_obj.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='runIdentification.py',
        description='''UT-Lift Script for Aerial Vehicles Identifying and Locating Markered Ground Crates'''
    )
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--hwil', action='store_false', help='in LIFT hwil mode')
    opt = parser.parse_args()

    # PREPARE DETECTION NETWORK MODEL
    print("[INFO] PREPARING DETECTION NETWORK MODEL ...")
    input_weights = opt.weights
    detectNN = dtctnn.detectionNetwork(input_weights)
    print("[INFO] DETECTION NETWORK MODEL PREPARED ...")

    # CLASSIFICATION FILTER AND ESTIMATION DON'T NEED TO BE PREPARED

    # HWIL SIM MODE
    if opt.hwil:
        runIdentifiaction_hwil()
    else:
        print('[ERROR]: ONLY HWIL SIM MODE IS READY')
