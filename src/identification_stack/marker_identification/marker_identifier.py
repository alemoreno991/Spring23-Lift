import numpy as np
import cv2 as cv

if __name__ == "__main__":
        import detection_nnetwork.detectc as dtctnn
        import classification_filter.imageClassifiers as cfilter
else:
        from .detection_nnetwork import detectc as dtctnn
        from .classification_filter import imageClassifiers as cfilter

class marker_identifier:
    
    classification  = None
    detection       = None
    cls_opt         = None
    is_dtct_opt     = False

    def __init__( self, cls_opt, dnn_weights = None ):

        self.cls_opt        = cls_opt
        self.is_dtct_opt    = not (dnn_weights is None)

        if( self.cls_opt == r'utlift' and self.is_dtct_opt  ):
            print("[INFO] PREPARING DETECTION NETWORK MODEL ...")
            self.detection      = dtctnn.detectionNetwork( dnn_weights )
            print("[INFO] DETECTION NETWORK MODEL PREPARED ...")
            self.classification = cfilter.utlift_classifier( True, 0 )

        elif( self.cls_opt == r'aruco' ):
            aruco_opts          = cfilter.aruco_options( cv.aruco.DetectorParameters_create(), cv.aruco.Dictionary_get( cv.aruco.DICT_4X4_50 ) )
            self.classification = cfilter.aruco_classifier( aruco_opts )


    def runIdentification( self, imgi ):
        
        if( self.cls_opt == r'utlift' and self.is_dtct_opt  ):
            # USE NN TO DETECT MARKER
            bbcrnrs_list            = self.detection.imageDetection(imgi)
            read_code_list          = []
            crate_center_pxlpt_list = []
            # CLASSIFY MARKERS IN IMAGE
            for bbcrnr in bbcrnrs_list:
                read_code                   = []
                crate_center_pxlpt          = []
                read_code, sctn_cntr_pxlpnt = self.classification.classifyImage( imgi, bbcrnr )
                x1, y1 = int(bbcrnr[0]), int(bbcrnr[1])
                
                if len(read_code) > 0:
                    if len( sctn_cntr_pxlpnt ) > 0:
                        crate_center_pxlpt  = np.transpose( np.array( [ x1, y1 ] ) + sctn_cntr_pxlpnt )
                    read_code_list.append( read_code )
                    crate_center_pxlpt_list.append( crate_center_pxlpt )
                else:
                    continue
            
            return read_code_list, crate_center_pxlpt_list
        
        elif( self.cls_opt == r'aruco' ):
            read_code_list, crate_center_pxlpt_list = self.classification.classifyImage( imgi )
            print('\n[RESULT]: ' + str(read_code_list))
            return read_code_list, crate_center_pxlpt_list

        
        return [],[]