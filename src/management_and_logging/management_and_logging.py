import argparse
import csv 

import cv2 as cv
import numpy as np

# SERIAL CONNECTION ICP PACKAGES
import zmq
import struct

if __name__ == "__main__":
        import utils.mnlConst as mnlc 
else:
        from .utils import mnlConst as mnlc 

###############################################################################

class dataStreamManager:
    struct_format   = mnlc.PYSTRUCT_PACKETFORMAT
    ipc_topic_enc   = mnlc.IPC_TOPIC.encode('ascii')
    sub_context     = zmq.Context()
    sub_socket      = sub_context.socket(zmq.SUB)
    sub_addr        = None

    def __init__( self, sub_address_string ) -> None:
        if not (sub_address_string is None): 
            # TELEMETRY DATA LISTENER 
            self.sub_socket.setsockopt( zmq.SUBSCRIBE, self.ipc_topic_enc )
            self.sub_socket.setsockopt( zmq.CONFLATE, 1 )
            self.sub_socket.connect( str(sub_address_string) )
            self.sub_addr = sub_address_string
        else:
            print("[INFO]: DATA STREAM OPTION WASN'T TURNED ON ... NOT STREAMING")
        
    def __del__(self):
        if not (self.sub_addr is None): 
            # CLOSE CONNECTION TO TELEMETRY 
            self.sub_socket.disconnect( self.sub_addr )
            self.sub_socket.close()
            self.sub_context.term()

    def get_dataPackage(self):
        if not (self.sub_addr is None): 
            try:
                _ , data_buffer = self.sub_socket.recv().split(b' ', 1)
                data = struct.unpack(self.struct_format, data_buffer)
                return data
            except:
                return []
        else:
            return []

###############################################################################

class captureStreamManager:
    cap_obj = None

    def __init__( self, pipleline, cap_opt = cv.CAP_GSTREAMER ):
        if not (pipleline is None): 
            print('[INFO]: ATTEMPTING TO OPEN CAPTURE OBJECT')
            self.cap_obj = cv.VideoCapture( pipleline , cap_opt )
            if not self.cap_obj.isOpened():
                print('[ERROR]: CAPTURE OBJECT DID NOT OPEN ... EXITING')
                exit(0)
            else:
                print('[INFO]: CAPTURE OBJECT IS OPEN.')
        else:
            print("[INFO]: VIDEO CAPTURE OPTION WASN'T TURNED ON ... NOT STREAMING")

    def __del__( self ):
        if not (self.cap_obj is None): 
            self.cap_obj.release()

    def get_Image( self ):
        if not (self.cap_obj is None): 
            try:
                _, imgi = self.cap_obj.read()
                return imgi
            except:
                return []
        else:
            return []

###############################################################################

class dataLogger:
    fout            = None
    csv_out         = None
    i               = 0

    def __init__( self, file_string ) -> None:
        # DATA WRITER TO CSV
        if not (file_string is None): 
            self.fout    = open( file_string ,'w')
            self.csv_out = csv.writer( self.fout )
        else:
            print("[INFO]: FILE SAVING OPTION WASN'T TURNED ON ... NOT SAVING")
        
    def __del__(self):
        
        # CLOSE CONNECTION TO FILE
        if not (self.fout is None):
            self.fout.close()

    def save_data( self, data, imgi ):
        if not (self.fout is None):
            try:
                self.csv_out.writerow( data )
                cv.imwrite( "./data/images/img_{}.jpg".format(self.i), imgi )
                self.i += 1
            except:
                return 
        
###############################################################################