import cv2 as cv
import math
import numpy as np

if __name__ == "__main__":
        import utils.decodeConst as dcdc
        import utils.drawingTools as dt
        import utils.metrics2D as est
        import utils.decodeTools as dcdt
else:
        from .utils import decodeConst as dcdc 
        from .utils import drawingTools as dt
        from .utils import metrics2D as est
        from .utils import decodeTools as dcdt
         
        
import time
import sys

# TODO ADD CHECK FOR WHEN ID TOO MANY OR TOO LITTLE BARS WERE FOUND
def extractD1Domain(image, debug_mode, on_hardware = False):

        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-dcdc.GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        src_blur = cv.GaussianBlur(255-src_thresh_prelim,( int( 2*dcdc.GAUSS_BLUR_MORPH_SIZE+1 ), int( 2*dcdc.GAUSS_BLUR_MORPH_SIZE+1 ) ), 0 )
        src_eval_contours = cv.medianBlur( src_blur, int( 2*dcdc.MEDIAN_BLUR_SIZE+1 ) )

        #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END
        if(debug_mode and (not on_hardware) ):
                cv.imshow( "D3 IMAGE USED TO CALCULATE CONTOURS", src_eval_contours) 
                cv.waitKey(0)

        #  FIND LOCATING BARS - START
        rect_contours, rect_contour_centroids, rect_contour_angles, rect_contour_areas, poly_contour_areas = \
                dcdt.determineCandidateRectIDbars( src_eval_contours, debug_mode, on_hardware )

        if( debug_mode and len(rect_contour_centroids) < 2 ):
                print("\n[DEBUG]: CLASSIFICATION FILTER FOUND LESS THAN 2 ID-BARS ... SKIPPING")
                return [],dcdt.cid_information(-1,-1,False),[],[]

        #  FIND LOCATING BARS - END

        #########################################################################################################################################
        #  PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - START
        if(debug_mode):
                print( "\n[DEBUG]: FOUND: " + str(len(rect_contour_centroids)) + " IDENTIFING BARS" )
                if( not on_hardware ):
                        dt.showContours(rect_contours,"CONTOURS THAT PASS ALL GEOMETRY TESTS",image.shape)
                        cv.waitKey(0)


        rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas = \
                dcdt.attemptIdBarCorrections( image.shape, rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas, debug_mode, on_hardware )

        if( len(rect_contour_centroids) > 4 or len(rect_contour_centroids) < 2 ):
                if(debug_mode):
                        print( "\n[DEBUG]: ID BAR CORRECTION FAILED. THERE ARE " + str(len(rect_contour_centroids)) + " ID-BARS ... SKIPPING"  )
                return [],dcdt.cid_information(-1,-1,False),[],[]
        else:
                if(debug_mode):
                        print( "\n[DEBUG]: ID BAR CORRECTION SUCCESS. THERE ARE " + str(len(rect_contour_centroids)) + " ID-BARS" )

        if( dcdc.DECODER_SHOWCASE_MODE and (not on_hardware) ):
                dt.showContoursAndCentersOnImage(rect_contours,rect_contour_centroids,"PASSING CONTOURS ON IMAGE",image)
                cv.waitKey(0)
        #  PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - END
        #########################################################################################################################################


        warped_image = []
        encodedImage = []
        fc_median_distance = 0
        # EXTRACT WARPED IMAGE
        if( len(rect_contours) == 4 ):
                warped_image,fc_median_distance,transformation_mat = dcdt.determineWarpedImageFrom4IdBars( image, rect_contour_centroids, debug_mode ) 
                encodedImage = warped_image[ int(float(dcdc.ENCODING_CROP_RATIO)*fc_median_distance):int((1-float(dcdc.ENCODING_CROP_RATIO))*fc_median_distance), \
                                        int(float(dcdc.ENCODING_CROP_RATIO)*fc_median_distance):int((1-float(dcdc.ENCODING_CROP_RATIO))*fc_median_distance) ]
   
        elif( len(rect_contour_centroids) < 4 and len(rect_contour_centroids) > 1):
                
                corner_points = dcdt.determineD1Corners(src_eval_contours, rect_contours, rect_contour_centroids, rect_contour_angles, debug_mode, on_hardware)

                if len(corner_points) == 4:
                        encodedImage,fc_median_distance,transformation_mat = dcdt.determineWarpedImageFrom4Corners(image, corner_points, debug_mode)
                else:
                        return [],dcdt.cid_information(-1,-1,False),[],[]
        else: 
                return [],dcdt.cid_information(-1,-1,False),[],[]
        
        ######################################################################################################################################### POSE DETERMINATION - START
        rvec, tvec = [],[]
        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        img_height, img_width, _ = encodedImage.shape
        row_seg = int(img_height/dcdc.ENCODING_LENGTH)
        col_seg = int(img_width/dcdc.ENCODING_LENGTH)
        segment_area = img_height * img_width / ( dcdc.ENCODING_LENGTH * dcdc.ENCODING_LENGTH )
        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END

        ### # TODO: REARRANGE FUNCTION EXECUTION SO THIS DOESN'T HAPPEN TWICE
        encoded_gray = cv.cvtColor( encodedImage, cv.COLOR_BGR2GRAY )
        _,encoded_thresh_prelim = cv.threshold( 255-encoded_gray, 255-dcdc.GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        encoded_eval_contours = 255 - encoded_thresh_prelim
        ###

        cid_info = dcdt.determineCIDIndices( encoded_eval_contours, row_seg, col_seg, segment_area, debug_mode, on_hardware)
        if cid_info.found:
                        try:
                                rvec, tvec = dcdt.determinePose(transformation_mat, fc_median_distance, cid_info.corner_indx)
                                if( debug_mode and (not on_hardware)):  
                                        dt.showAxesOnImage(rvec, tvec, dcdc.CAMERA_MATRIX, dcdc.DISTANCE_COEFFICIENTS, "Axes", image)
                                        cv.waitKey(0)
                        except:
                                pass
        ######################################################################################################################################### POSE DETERMINATION - END

        if(debug_mode and (not on_hardware) and len(rect_contours) ==4):
                cv.imshow( "D2 CROPPED IMAGE", warped_image); 
                cv.imshow( "D1 ENCODED IMAGE", encodedImage); 
                cv.waitKey() 
        if(debug_mode and (not on_hardware) and len(rect_contour_centroids) < 4 and len(rect_contour_centroids) > 1):
                cv.imshow( "D1 ENCODED IMAGE", encodedImage); 
                cv.waitKey()
        return encoded_eval_contours, cid_info, rvec, tvec



def determineEncodingFromD1Image( image, cid_info, debug_mode, on_hardware = False ):
       
        # # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        # img_height, img_width, _ = image.shape
        # row_seg = int(img_height/dcdc.ENCODING_LENGTH)
        # col_seg = int(img_width/dcdc.ENCODING_LENGTH)
        # segment_area = row_seg * col_seg

        # src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        # _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-dcdc.GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        # src_eval_contours = 255 - src_thresh_prelim
        # #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END
        
        # # LOCATING CIRCULAR IDENTIFIER - START

        # cid_info.corner_indx, cid_info.indx, cid_info.found = dcdt.determineCIDIndices( src_eval_contours, row_seg, col_seg, segment_area, debug_mode, on_hardware )

        src_eval_contours = image
        img_height, img_width = image.shape
        row_seg = int(img_height/dcdc.ENCODING_LENGTH)
        col_seg = int(img_width/dcdc.ENCODING_LENGTH)
        segment_area = row_seg * col_seg
        if( not cid_info.found ):
                return []
        # LOCATING CIRCULAR IDENTIFIER - END


        # DETERMINE BIT ENCODING - START
        if dcdc.ENCODING_VERSION == 1:
                pre_bit_encoding, pre_bit_pass = dcdt.evaluateV1BitEncoding( src_eval_contours, row_seg, col_seg, segment_area, cid_info.indx, debug_mode, on_hardware )
        elif dcdc.ENCODING_VERSION == 2:
                pre_bit_encoding, pre_bit_pass = dcdt.evaluateV2BitEncoding( src_eval_contours, row_seg, col_seg, segment_area, cid_info.indx, cid_info.corner_indx, debug_mode, on_hardware )
        else:
                print( "\n[ERROR]: LATEST ENCODING VERSION IS v2. VERSION REQUESTED WAS: v" + str(dcdc.ENCODING_VERSION) )
                return []
        
        bit_encoding = []
        if( pre_bit_pass and cid_info.found):
                bit_encoding =  dcdt.readMappedEncoding( cid_info.indx, pre_bit_encoding )

        # DETERMINE BIT ENCODING - END
        if(dcdc.DECODER_SHOWCASE_MODE and (not on_hardware)):
                show_image = src_eval_contours.copy()
                dt.showEncodingInformation( pre_bit_encoding, row_seg, dcdc.ENCODING_LENGTH, "DECODED MESSAGE",  show_image ) 
                cv.waitKey(0)

        return bit_encoding



def decodeImage(image, cascade_debug_mode, cascade_on_hardware = False):
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
        encodedImage, cid_info, _, _ = extractD1Domain(image, DEBUG_D1E_FLAG, cascade_on_hardware )
        if not cid_info.found:
                return []
        if not len(encodedImage): 
                print('\n[RESULT]: ENCODED ROI COULD NOT BE DETERMINED')
                return []
        else:
                bit_encoding = determineEncodingFromD1Image( encodedImage, cid_info, DEBUG_DENC_FLAG, cascade_on_hardware )
                if not len(bit_encoding):
                        print('\n[RESULT]: MARKER ENCODING COULD NOT BE DECODED')
                else:
                        print('\n[RESULT]: ' + str(bit_encoding))
                
                return bit_encoding



def decodeImageSection(image, crnrs, cascade_debug_mode, cascade_on_hardware = False):
        imgn = image
        x1, y1, x2, y2 = int(crnrs[0]), int(crnrs[1]), int(crnrs[2]), int(crnrs[3])
        image_section = imgn[y1:y2, x1:x2]

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
        encodedImage, cid_info, _, _ = extractD1Domain(image_section, DEBUG_D1E_FLAG, cascade_on_hardware )
        if not cid_info.found:
                return []
        if not len(encodedImage): 
                print('\n[RESULT]: ENCODED ROI COULD NOT BE DETERMINED')
                return []
        else:
                bit_encoding = determineEncodingFromD1Image( encodedImage, cid_info, DEBUG_DENC_FLAG, cascade_on_hardware )
                if not len(bit_encoding):
                        print('\n[RESULT]: MARKER ENCODING COULD NOT BE DECODED')
                else:
                        print('\n[RESULT]: ' + str(bit_encoding))
                
                return bit_encoding

        

def main():
        imageName = str( sys.argv[1] )

        start = time.time()
        src = cv.imread(imageName)
        assert imageName is not None, "[ERROR] file could not be read, check with os.path.exists()"
        bitencoding = decodeImage(src,dcdc.DECODER_DEBUG_MODE)

        print("\nPREDICTED ENCODING:\n"+str(bitencoding))
        print("Process Time: " + str( round( time.time() - start, 5 )  ) )

if __name__ == '__main__':
        main()
