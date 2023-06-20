import cv2 as cv
import math
import numpy as np
from . import decodeConst as dcdc
from . import drawingTools as dt
from . import metrics2D as est
from . import decodeTools as dcdt
import time
import sys


def extractD1Domain(image, debug_mode):

        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-dcdc.GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        src_blur = cv.GaussianBlur(255-src_thresh_prelim,( int( 2*dcdc.GAUSS_BLUR_MORPH_SIZE+1 ), int( 2*dcdc.GAUSS_BLUR_MORPH_SIZE+1 ) ), 0 )
        src_eval_contours = cv.medianBlur( src_blur, int( 2*dcdc.MEDIAN_BLUR_SIZE+1 ) )

        #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END
        if(debug_mode):
                cv.imshow( "D3 IMAGE USED TO CALCULATE CONTOURS", src_eval_contours) 
                cv.waitKey(0)

        #  FIND LOCATING BARS - START
        rect_contours, rect_contour_centroids, rect_contour_angles, rect_contour_areas, poly_contour_areas = \
                dcdt.determineCandidateRectIDbars( src_eval_contours, debug_mode )

        if( len(rect_contour_centroids) < 2 ):
                print("[ERROR] CLASSIFICATION FILTER FOUND LESS THAN 2 ID-BARS AND CAN NOT CONTINUE")
                return []

        #  FIND LOCATING BARS - END

        #########################################################################################################################################
        #  PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - START
        if(debug_mode):
                print( "[DEBUG] FOUND: " + str(len(rect_contour_centroids)) + " IDENTIFING BARS" )
                dt.showContours(rect_contours,"CONTOURS THAT PASS ALL GEOMETRY TESTS",image.shape)
                cv.waitKey(0)


        rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas = \
                dcdt.attemptIdBarCorrections( rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas, debug_mode, image.shape )

        if( len(rect_contour_centroids) > 4 or len(rect_contour_centroids) < 2 ):
                if(debug_mode):
                        print( "[ERROR] ID BAR CORRECTION FAILED AS THERE ARE >4 ID-BARS OR >2 ID-BARS: " + str(len(rect_contour_centroids))  )
                return []
        else:
                if(debug_mode):
                        print( "[DEBUG] ID BAR CORRECTION SUCCESS AS THERE ARE <4 ID-BARS: " + str(len(rect_contour_centroids))  )

        if(dcdc.DECODER_SHOWCASE_MODE):
                dt.showContoursAndCentersOnImage(rect_contours,rect_contour_centroids,"PASSING CONTOURS ON IMAGE",image)
                cv.waitKey(0)
        #  PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - END
        #########################################################################################################################################


        warped_image = []
        fc_median_distance = 0.
        if( len(rect_contour_centroids) == 4 ):
                warped_image,fc_median_distance = dcdt.determineWarpedImageFrom4IdBars( image, rect_contour_centroids, debug_mode )
        elif( len(rect_contour_centroids) < 4 and len(rect_contour_centroids) > 1):
                warped_image,fc_median_distance = dcdt.determineWarpedImageFrom2or3IdBars( image, rect_contour_centroids, rect_contour_angles, debug_mode)
        else: 
                return []

        encodedImage = warped_image[ int(float(dcdc.ENCODING_CROP_RATIO)*fc_median_distance):int((1-float(dcdc.ENCODING_CROP_RATIO))*fc_median_distance), \
                                     int(float(dcdc.ENCODING_CROP_RATIO)*fc_median_distance):int((1-float(dcdc.ENCODING_CROP_RATIO))*fc_median_distance) ]

        if(debug_mode):
                cv.imshow( "D2 CROPPED IMAGE", warped_image); 
                cv.imshow( "D1 ENCODED IMAGE", encodedImage); 
                cv.waitKey(); 
        
        return encodedImage



def determineEncodingFromD1Image( image, debug_mode ):
        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        img_height, img_width, _ = image.shape
        row_seg = int(img_height/dcdc.ENCODING_LENGTH)
        col_seg = int(img_width/dcdc.ENCODING_LENGTH)
        segment_area = img_height * img_width / ( dcdc.ENCODING_LENGTH * dcdc.ENCODING_LENGTH )

        src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-dcdc.GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        src_eval_contours = 255 - src_thresh_prelim
        #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END
        
        # LOCATING CIRCULAR IDENTIFIER - START
        cid_corner_indx, cid_indx, cid_found = dcdt.determineCIDIndices( src_eval_contours, row_seg, col_seg, segment_area, debug_mode )
        if( not cid_found ):
                return []
        # LOCATING CIRCULAR IDENTIFIER - END


        # DETERMINE BIT ENCODING - START
        if dcdc.ENCODING_VERSION == 1:
                pre_bit_encoding, pre_bit_pass = dcdt.evaluateV1BitEncoding( src_eval_contours, row_seg, col_seg, segment_area, cid_indx, debug_mode )
        elif dcdc.ENCODING_VERSION == 2:
                pre_bit_encoding, pre_bit_pass = dcdt.evaluateV2BitEncoding( src_eval_contours, row_seg, col_seg, segment_area, cid_indx, cid_corner_indx, debug_mode )
        else:
                print( "LATEST ENCODING VERSION IS v2. VERSION REQUESTED WAS: v" + str(dcdc.ENCODING_VERSION) )
                return []
        
        bit_encoding = []
        if( pre_bit_pass and cid_found):
                bit_encoding =  dcdt.readMappedEncoding( cid_indx, pre_bit_encoding )


        # DETERMINE BIT ENCODING - END
        if(dcdc.DECODER_SHOWCASE_MODE):
                show_image = image.copy()
                dt.showEncodingInformation( pre_bit_encoding, row_seg, dcdc.ENCODING_LENGTH, "DECODED MESSAGE",  show_image ) 
                cv.waitKey(0)

        return bit_encoding



def decodeImage(image, cascade_debug_mode):
        DEBUG_D1E_FLAG = False
        DEBUG_DENC_FLAG = False
        if(cascade_debug_mode == 1):
                DEBUG_D1E_FLAG = True
                DEBUG_DENC_FLAG = False
        elif(cascade_debug_mode == 2):
                DEBUG_D1E_FLAG = False
                DEBUG_DENC_FLAG = True
        elif(cascade_debug_mode == 3):
                DEBUG_D1E_FLAG = True
                DEBUG_DENC_FLAG = True
        
        encodedImage = extractD1Domain(image, DEBUG_D1E_FLAG)
        if not len(encodedImage): 
                return []
        else:
                return determineEncodingFromD1Image( encodedImage, DEBUG_DENC_FLAG )


def decodeImageSection(image, crnrs):
        imgn = image
        x1, y1, x2, y2 = int(crnrs[0]), int(crnrs[1]), int(crnrs[2]), int(crnrs[3])
        image_section = imgn[y1:y2, x1:x2]
        encodedImage = extractD1Domain( image_section, False )
        
        if not len(encodedImage): 
                return []
        else:
                return determineEncodingFromD1Image( encodedImage, False )

        

def main():
        imageName = str( sys.argv[1] )

        start = time.time()
        src = cv.imread(imageName)
        assert imageName is not None, "[ERROR] file could not be read, check with os.path.exists()"
        bitencoding = decodeImage(src,dcdc.DECODER_DEBUG_MODE)

        print("\nPREDICTED ENCODING:\n"+str(bitencoding)+'\n')
        print("Process Time: " + str( round( time.time() - start, 5 )  ) + '\n')

if __name__ == '__main__':
        main()
