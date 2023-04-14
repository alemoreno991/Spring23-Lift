import cv2 as cv
import math
import numpy as np
import decodeConst as dcdc
import drawingTools as dt
import metrics2D as est
import decodeTools as dcdt
import sys


def extractD1Domain(image, debug_mode):

        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        img_height, img_width, _ = image.shape
        src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-dcdc.GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        # src_blur = cv.medianBlur( 255-src_thresh_prelim, int( 2*dcdc.MEDIAN_BLUR_SIZE+1 ) )
        src_blur = cv.GaussianBlur(255-src_thresh_prelim,(5,5),0)
        element1 = cv.getStructuringElement( 2, \
                                             ( int( 2*dcdc.CLOSING_MORPH_SIZE+1 ), int( 2*dcdc.CLOSING_MORPH_SIZE+1 ) ), \
                                             ( int( dcdc.CLOSING_MORPH_SIZE ), int( dcdc.CLOSING_MORPH_SIZE ) ) \
                                           )

        src_eval_contours = cv.morphologyEx( src_blur, cv.MORPH_CLOSE, element1)
        #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END

        #  FIND LOCATING BARS - START
        rect_contours, rect_contours_pass1, rect_contour_centroids, rect_contour_angles, rect_contour_areas, poly_contour_areas = [], [], [], [], [], []
        canny_output = cv.Canny( src_eval_contours, 10, 200, True )
        contours, hierarchy = cv.findContours( canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS )
        for ii in range(0,len(contours)):
                approx_cp = cv.approxPolyDP( contours[ii], 0.025*cv.arcLength(contours[ii],True), True )
                approx_cp_area = abs( cv.contourArea(approx_cp) )
                rect_contours_pass1.append( contours[ii] )
                if( (     approx_cp_area > img_height * img_width * pow(float(dcdc.RECT_AREA_PERCENT_THRESHOLD)/100,2) ) \
                      and len(approx_cp) >= dcdc.RECT_IDENTIFIER_SIDES_LOWER_THRESHOLD \
                      and len(approx_cp) <= dcdc.RECT_IDENTIFIER_SIDES_UPPER_THRESHOLD  ):
                        approx_rect_center, approx_rect_size, approx_rect_angle  = cv.minAreaRect(approx_cp)
                        approx_aspectr = max( approx_rect_size[0]/approx_rect_size[1], approx_rect_size[1]/approx_rect_size[0] )

                        if(    approx_aspectr > (1 - float(dcdc.RECT_ASPECT_RATIO_LOWER_PERECNT_ERROR_THRESHOLD)/100) * dcdc.MEASURED_ASPECT_RATIO 
                           and approx_aspectr < (1 + float(dcdc.RECT_ASPECT_RATIO_UPPER_PERECNT_ERROR_THRESHOLD)/100) * dcdc.MEASURED_ASPECT_RATIO ):
                                rect_contour_areas.append( approx_rect_size[0]*approx_rect_size[1] )
                                poly_contour_areas.append( approx_cp_area ) # NOT THE SAME AS THE ONE ABOVE
                                rect_contours.append( contours[ii] )
                                rect_contour_centroids.append(approx_rect_center)
                                if( approx_rect_size[0] < approx_rect_size[1] ):
                                        rect_contour_angles.append(approx_rect_angle)
                                else:
                                        rect_contour_angles.append( 90. - approx_rect_angle )

        if( len(rect_contour_centroids) < 2 ):
                print("[ERROR] CLASSIFICATION FILTER FOUND LESS THAN 2 ID-BARS AND CAN NOT CONTINUE")
                return []

        if(debug_mode):
                cv.imshow( "D3 IMAGE USED TO CALCULATE CONTOURS", src_eval_contours) 
                dt.showContours(contours,"CONTOURS IN D3",image.shape)
                dt.showContours(rect_contours_pass1,"CONTOURS THAT PASS EVERYTHING BUT ASPECT-RATIO",image.shape)
                cv.waitKey(0)
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
                print( "[DEBUG] ID BAR CORRECTION SUCCESS AS THERE ARE <4 ID-BARS: " + str(len(rect_contour_centroids))  )

        if(dcdc.DECODER_SHOWCASE_MODE):
                dt.showContoursAndCentersOnImage(rect_contours,hierarchy,rect_contour_centroids,"PASSING CONTOURS ON IMAGE",image)
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
                pre_bit_encoding, pre_bit_pass = dcdt.evaluateV1BitEncoding( )
        elif dcdc.ENCODING_VERSION == 2:
                pre_bit_encoding, pre_bit_pass = dcdt.evaluateV2BitEncoding( src_eval_contours, row_seg, col_seg, segment_area, cid_indx, cid_corner_indx, debug_mode )
        else:
                return []
        
        bit_encoding = []
        if( pre_bit_pass and cid_found):
                bit_encoding =  dcdt.readMappedEncoding( cid_indx, pre_bit_encoding )

        # DETERMINE BIT ENCODING - END
        if(dcdc.DECODER_SHOWCASE_MODE):
                show_image = image
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
        

def main():
        imageName = str( sys.argv[1] )
        src = cv.imread(imageName)
        assert imageName is not None, "[ERROR] file could not be read, check with os.path.exists()"
        bitencoding = decodeImage(src,dcdc.DECODER_DEBUG_MODE)
        print(bitencoding)
        

if __name__ == '__main__':
        main()
