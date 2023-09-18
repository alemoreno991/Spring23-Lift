import deploymentConst as dplyC
import cv2 as cv
import time

def main():
    
    cap_obj = cv.VideoCapture( 'udpsrc port=1234 ! video/mpegts ! tsdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false drop=true max-buffers=1' , cv.CAP_GSTREAMER )
    
    if not cap_obj.isOpened():
        print('[ERROR]: CAPTURE OBJECT DID NOT OPEN ... EXITING')
        exit(0)
    else:
        print('[DEBUG]: CAPTURE OBJECT OPENNED. STARTING READ PROCESS')

    for ii in range(0,6):
        _,frame = cap_obj.read()
        print('[DEBUG] NUMBER OF IMAGE COLOR CHANNELS: {}'.format( frame.shape[-1] ) )
        cv.imwrite('./gscnnct_bin/test_im{}.png'.format(ii),frame)
        print('[DEBUG]: IMAGE {} SAVED TO "./gscnnct_bin/". WAITING 5 SECONDS'.format(ii))
        time.sleep(5)

    print('[RESULT]: VIDEOCAPTURE TEST COMPLETED CHECK "./gscnnct_bin/"')
    cap_obj.release()




if __name__ == "__main__":
    main()
