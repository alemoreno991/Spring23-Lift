import deploymentConst as dplyC
import cv2 as cv
import time

def main():
    
    cap_obj = None
    if dplyC.CAPTURE_TYPE == 0:
        cap_obj = rtrv_gs_cnct()
    else:
        print("[ERROR]: INVALID CAPTURE TYPE")
    
    if not cap_obj.isOpened():
        print('[ERROR]: CAPTURE OBJECT DID NOT OPEN ... EXITING')
        exit(0)
    else:
        print('[DEBUG]: CAPTURE OBJECT OPENNED. STARTING READ PROCESS')

    for ii in range(0,6):
        _,frame = cap_obj.read()
        cv.imwrite('./gscnnct_bin/test_im{}.png'.format(ii),frame)
        print('[DEBUG]: IMAGE {} SAVED TO "./gscnnct_bin/". WAITING 5 SECONDS'.format(ii))
        time.sleep(5)

    print('[RESULT]: VIDEOCAPTURE TEST COMPLETED CHECK "./gscnnct_bin/"')
    cap_obj.release()


def rtrv_gs_cnct():
    gspline = 'udpsrc port={} caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! nvv4l2decoder ! videoconvert ! appsink'.format(dplyC.NTWRK_RTRVL_PORT)
    gscap = cv.VideoCapture( gspline, cv.CAP_GSTREAMER )
    return gscap


if __name__ == "__main__":
    main()
