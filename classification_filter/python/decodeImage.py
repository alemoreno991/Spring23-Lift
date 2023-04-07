import cv2 as cv
import math
import numpy as np
import debugTools as dt
import euclideanSpaceTools as est
import sys

#// DEBUG PROPERTIES
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DECODER_SHOWCASE_MODE = True
DECODER_DEBUG_MODE = 0
#  0 - NONE
#  1 - DEBUGGER MODE FOR D1 EXTRACTION
#  2 - DEBUGGER MODE FOR ENCODING DETERMINATION
#  3 - DEBUGGER MODE FOR BOTH MODES (1)(2)
DECODER_VERBOSE_MODE = False
DECODER_SHOWCASE_MODE = True
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#/// MEASURED PROPERTIES 
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#// // FOR COMPUTER GENERATED IMAGES
# ENCODING_LENGTH 3
# MEASURED_ASPECT_RATIO 9.615
# ENCODING_CROP_RATIO 0.1772151899
# SEGMENT_LCIRC_RATIO 0.7853981634
# SEGMENT_SCIRC_RATIO 0.0872664626

#// // FOR IRL IMAGES
ENCODING_LENGTH = 3
MEASURED_ASPECT_RATIO = 6.0
ENCODING_CROP_RATIO = 0.1666666667
SEGMENT_LCIRC_RATIO = 0.7853981634
SEGMENT_SCIRC_RATIO = 0.0872664626
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#/// TUNING PARAMETERS
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GREYSCALE_TO_255_THRESHOLD = 200

RECT_AREA_PERCENT_THRESHOLD = 0.15
RECT_ASPECT_RATIO_LOWER_PERECNT_ERROR_THRESHOLD = 20
RECT_ASPECT_RATIO_UPPER_PERECNT_ERROR_THRESHOLD = 40
RECT_IDENTIFIER_SIDES_LOWER_THRESHOLD = 4
RECT_IDENTIFIER_SIDES_UPPER_THRESHOLD = 6

CONTOUR_ED_THRES = 20

CIRC_AREA_LOWER_PERCENT_THRESHOLD = 30
CIRC_AREA_UPPER_PERCENT_THRESHOLD = 10
CIRC_IDENTIFIER_SIDES_THRESHOLD = 10

DECODING_CONFIDENCE_THRESHOLD = 0.65
DECODING_GREYSCALE_THRESH = 150

#// MORPHING PROPERTIES
GRADIENT_MORPH_SIZE = 1
CLOSING_MORPH_SIZE = 0
MEDIAN_BLUR_SIZE = 1
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def determineWarpedImageFrom4IdBars(image, rect_contour_centroids, debug_mode):
        fc_relative_angles = []
        quadrant0_flag = False
        quadrant3_flag = False
        for ii in range(1,4):
                anglei = est.angleBetweenPoints(rect_contour_centroids[0],rect_contour_centroids[ii])
                if( est.determineAngleQuadrant(anglei) == 0 ):
                        quadrant0_flag = True
                elif( est.determineAngleQuadrant(anglei) == 3 ):
                        quadrant3_flag = True
                fc_relative_angles.append(anglei)
        
        if( quadrant0_flag and quadrant3_flag ):
                for ii in range(0,3):
                        anglei = fc_relative_angles[ii] + math.pi; 
                        if( anglei > 2*math.pi ):
                                anglei = anglei - 2*math.pi
                        fc_relative_angles[ii] = anglei

        fc_ra_max = fc_relative_angles[0]
        fc_ra_min = fc_ra_max
        fc_ra_max_indx = int(1)
        fc_ra_min_indx = int(1)

        for ii in range(2,4):
                anglei = fc_relative_angles[ii-1]
                if( anglei < fc_ra_min ):
                        fc_ra_min = anglei
                        fc_ra_min_indx = ii
                elif( anglei > fc_ra_max ): 
                        fc_ra_max = anglei
                        fc_ra_max_indx = ii

        fc_median_indx = int(0)
        for ii in range(2,4):
            if( ii != fc_ra_min_indx and ii != fc_ra_max_indx ):
                fc_median_indx = ii
        
        fc_median_distance =  est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[fc_median_indx])

        ordered_centroids = np.array( [ [ rect_contour_centroids[0], rect_contour_centroids[fc_ra_min_indx], \
                                          rect_contour_centroids[fc_median_indx], rect_contour_centroids[fc_ra_max_indx] ] ] , \
                                      dtype = "float32" )
        
        perspective_transformed_centroids = np.array( [ [ ( 0.0, 0.5*fc_median_distance), ( 0.5*fc_median_distance, 0.0), \
                                                          ( fc_median_distance, 0.5*fc_median_distance), ( 0.5*fc_median_distance, fc_median_distance) ] ], \
                                                          dtype = "float32" )
        
        transformationMat = cv.getPerspectiveTransform(ordered_centroids, perspective_transformed_centroids)
        warped_image = cv.warpPerspective(image, transformationMat, (int(fc_median_distance), int(fc_median_distance)))
        
        return warped_image,fc_median_distance


def determineWarpedImageFrom2or3IdBars(image, rect_contour_centroids, rect_contour_angles, DEBUG_MODE):
        # TODO: CHECK THIS
        ref_angle = rect_contour_angles[0]
        slope = math.tan ( ref_angle * math.pi / 180.0 )
        yinter = rect_contour_centroids[0][1] - slope*rect_contour_centroids[0][0]
        above_reference_line = rect_contour_centroids[1][1] - (slope*rect_contour_centroids[1][0] + yinter) > 0

        fc_median_distance = 0.
        if( abs(ref_angle - rect_contour_angles[0]) > 45.0 ):
                fc_median_distance = 2 * est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1]) / math.sqrt(2)
        else:
                fc_median_distance = est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1])
        delta_angle = 0

        if(above_reference_line):
                delta_angle = -90 - ref_angle
        else:
                delta_angle = 90 - ref_angle

        img_height, img_width, _ = image.shape
        rotatedimageSize = ( int(2*img_height) , int(2*img_width)  )

        transformationMat = cv.getRotationMatrix2D( rect_contour_centroids[0], -delta_angle, 1 )
        rotatedImage = cv.warpAffine(image, transformationMat, rotatedimageSize )
        warped_image = rotatedImage[ int(rect_contour_centroids[0][1]-0.5*fc_median_distance):int( rect_contour_centroids[0][1]+0.5*fc_median_distance ) , \
                                     int(rect_contour_centroids[0][0]):int(rect_contour_centroids[0][0]+fc_median_distance ) ]
        
        return warped_image, fc_median_distance


def extractD1Domain(image, debug_mode):

        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        img_height, img_width, _ = image.shape
        src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        src_blur = cv.medianBlur( 255-src_thresh_prelim, int( 2*MEDIAN_BLUR_SIZE+1 ) )
        element1 = cv.getStructuringElement( 2, \
                                             ( int( 2*CLOSING_MORPH_SIZE+1 ), int( 2*CLOSING_MORPH_SIZE+1 ) ), \
                                             ( int( CLOSING_MORPH_SIZE ), int( CLOSING_MORPH_SIZE ) ) \
                                           )

        src_eval_contours = cv.morphologyEx( src_blur, cv.MORPH_CLOSE, element1)
        #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END

        #  FIND LOCATING BARS - START
        rect_contours = []
        rect_contours_pass1 = []
        rect_contour_centroids = []
        rect_contour_angles = []
        canny_output = cv.Canny( src_eval_contours, 10, 200 )
        contours, hierarchy = cv.findContours( canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS )
        for ii in range(0,len(contours)):
                approx_cp = cv.approxPolyDP( contours[ii], 0.025*cv.arcLength(contours[ii],True), True )
                approx_cp_area = abs( cv.contourArea(approx_cp) )
                if( (    approx_cp_area > img_height * img_width * float(RECT_AREA_PERCENT_THRESHOLD)/100 ) \
                     and len(approx_cp) >= RECT_IDENTIFIER_SIDES_LOWER_THRESHOLD \
                     and len(approx_cp) <= RECT_IDENTIFIER_SIDES_UPPER_THRESHOLD  ):
                        rect_contours_pass1.append( contours[ii] )
                        _, approx_rect_size, approx_rect_angle  = cv.minAreaRect(approx_cp)
                        approx_aspectr = max( approx_rect_size[0]/approx_rect_size[1], approx_rect_size[1]/approx_rect_size[0] )
                        if(    approx_aspectr > (1 - float(RECT_ASPECT_RATIO_LOWER_PERECNT_ERROR_THRESHOLD)/100) * MEASURED_ASPECT_RATIO 
                           and approx_aspectr < (1 + float(RECT_ASPECT_RATIO_UPPER_PERECNT_ERROR_THRESHOLD)/100) * MEASURED_ASPECT_RATIO ):
                                rect_contours.append( contours[ii] )
                                contour_moments = cv.moments( contours[ii] )
                                contour_centroid = ( (contour_moments['m10']/contour_moments['m00']) , (contour_moments['m01']/contour_moments['m00']) )
                                rect_contour_centroids.append(contour_centroid)
                                if( approx_rect_size[0]/approx_rect_size[1] < 1 ):
                                        rect_contour_angles.append(approx_rect_angle)
                                else:
                                        rect_contour_angles.append( -np.sign(approx_rect_angle) * ( 90.-abs(approx_rect_angle) ))

        if(debug_mode):
                cv.imshow( "D3 IMAGE USED TO CALCULATE CONTOURS", src_eval_contours) 
                dt.showContours(contours,hierarchy,"CONTOURS IN D3",image)
                dt.showContours(rect_contours_pass1,hierarchy,"CONTOURS THAT PASS EVERYTHING BUT ASPECT-RATIO",image)
                cv.waitKey(0)
        #  FIND LOCATING BARS - END

        if(DECODER_SHOWCASE_MODE):
                dt.showContoursAndCentersOnImage(rect_contours,hierarchy,rect_contour_centroids,"PASSING CONTOURS ON IMAGE",image)
                cv.waitKey(0)

        #  PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - START
        rect_contours_corr = []
        rect_contour_centroids_corr = []
        rect_contour_angles_corr = []
        if( len(rect_contour_centroids) > 4 ):
                if(debug_mode):
                        print( "[WARNING] FOUND: " + str(len(rect_contour_centroids)) + " IDENTIFING BARS" )
                        dt.showContours(rect_contours,hierarchy,"CONTOURS THAT PASS ALL GEOMETRY TESTS",image)
                        cv.waitKey(0)

                rect_contours_corr.append( rect_contours[0] )
                rect_contour_centroids_corr.append( rect_contour_centroids[0] )
                rect_contour_angles_corr.append( rect_contour_angles[0] )

                for ii in range( 1, len(rect_contour_centroids) ):
                        pass_flag = True
                        for jj in range( 0, len(rect_contour_centroids_corr) ):
                                if( est.euclideanDistance( rect_contour_centroids[ii], rect_contour_centroids_corr[jj] ) <= float(CONTOUR_ED_THRES) ):
                                        pass_flag = False 
                                        break
                        if( pass_flag ):
                                rect_contours_corr.append( rect_contours[ii] )
                                rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
                                rect_contour_angles_corr.append( rect_contour_angles[ii] )
                
                rect_contours = rect_contours_corr
                rect_contour_centroids = rect_contour_centroids_corr
                rect_contour_angles = rect_contour_angles_corr

                if( len(rect_contour_centroids) > 4 ):
                        if(debug_mode):
                                print( "[ERROR] ID BAR CORRECTION FAILED AS THERE ARE >4 ID-BARS: " + str(len(rect_contour_centroids))  )
                        return []
                else:
                        if(debug_mode):
                                print( "[DEBUG] ID BAR CORRECTION SUCCESS AS THERE ARE <4 ID-BARS: " + str(len(rect_contour_centroids))  )
                                dt.showContours(rect_contours,hierarchy,"CONTOURS THAT PASS ALL GEOMETRY AND REDUCTION TESTS",image)
                                cv.waitKey(0)
        #  PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - END

        warped_image = []
        fc_median_distance = 0.
        if( len(rect_contour_centroids) == 4 ):
                warped_image,fc_median_distance = determineWarpedImageFrom4IdBars( image, rect_contour_centroids, debug_mode )
        elif( len(rect_contour_centroids) < 4 and len(rect_contour_centroids) > 1):
                warped_image,fc_median_distance = determineWarpedImageFrom2or3IdBars( image, rect_contour_centroids, rect_contour_angles, debug_mode)
        else: 
                return []

        encodedImage = warped_image[ int(float(ENCODING_CROP_RATIO)*fc_median_distance):int((1-float(ENCODING_CROP_RATIO))*fc_median_distance), \
                                     int(float(ENCODING_CROP_RATIO)*fc_median_distance):int((1-float(ENCODING_CROP_RATIO))*fc_median_distance) ]

        if(debug_mode):
                cv.imshow( "D2 CROPPED IMAGE", warped_image); 
                cv.imshow( "D1 ENCODED IMAGE", encodedImage); 
                cv.waitKey(); 
        
        return encodedImage


def determineEncodingFromD1Image( image, debug_mode ):
        bit_encoding = []
        # SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        img_height, img_width, _ = image.shape
        row_seg = int(img_height/ENCODING_LENGTH)
        col_seg = int(img_width/ENCODING_LENGTH)
        segment_area = img_height * img_width / ( ENCODING_LENGTH * ENCODING_LENGTH )

        src_gray = cv.cvtColor( image, cv.COLOR_BGR2GRAY )
        _,src_thresh_prelim = cv.threshold( 255-src_gray, 255-GREYSCALE_TO_255_THRESHOLD, 255, cv.THRESH_TOZERO )
        src_eval_contours = 255 - src_thresh_prelim
        #  SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END
        
        
        # LOCATING CIRCULAR IDENTIFIER - START
        canny_output = cv.Canny( src_eval_contours, 10, 200 )
        contours, hierarchy = cv.findContours( canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1 )
        weighted_sum_circ_centroid = (0.0, 0.0)
        total_weighted_area = 0.0

        circ_contours = []
        circ_contour_centroids = []
        for ii in range(0,len(contours)):
                approx_cp = cv.approxPolyDP( contours[ii], 0.01*cv.arcLength(contours[ii],True), True )
                if( len(approx_cp) > CIRC_IDENTIFIER_SIDES_THRESHOLD  ):
                        approx_min_circ_center,approx_min_circ_radius = cv.minEnclosingCircle( contours[ii] )
                        approx_cp_area = math.pi * pow(approx_min_circ_radius,2)
                        if(    approx_cp_area < ( segment_area ) * (SEGMENT_LCIRC_RATIO) * (1 + float(CIRC_AREA_UPPER_PERCENT_THRESHOLD)/100) \
                           and approx_cp_area > ( segment_area ) * (SEGMENT_SCIRC_RATIO) * (1 - float(CIRC_AREA_LOWER_PERCENT_THRESHOLD)/100)  ):
                                circ_contours.append( contours[ii] )
                                circ_contour_centroids.append( approx_min_circ_center )
                                weighted_sum_circ_centroid = ( weighted_sum_circ_centroid[0] + approx_cp_area*approx_min_circ_center[0], \
                                                               weighted_sum_circ_centroid[1] + approx_cp_area*approx_min_circ_center[1] ) 
                                total_weighted_area = total_weighted_area + approx_cp_area

        avg_circ_centroid = ( weighted_sum_circ_centroid[0]/total_weighted_area, weighted_sum_circ_centroid[1]/total_weighted_area  )
        # LOCATING CIRCULAR IDENTIFIER - END


        # DETERMINE BIT ENCODING - START
        cid_found = False
        cid_indx = int(-1) 
        encoded_src_threshed = src_eval_contours
        cidSegementSubmatrix = []
        for ii in range(0,ENCODING_LENGTH):
                if(cid_found):
                        break
                for jj in range(0,ENCODING_LENGTH):
                        if(cid_found):
                                break
                        elif( int(avg_circ_centroid[0]) >= jj*col_seg and int(avg_circ_centroid[0]) <= (jj+1)*col_seg - 1 \
                          and int(avg_circ_centroid[1]) >= ii*row_seg and int(avg_circ_centroid[1]) <= (ii+1)*row_seg - 1 ):
                                cid_indx = ii*ENCODING_LENGTH + jj
                                cid_found = True
                                cidSegementSubmatrix = encoded_src_threshed[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ]

        if( debug_mode ):
                dt.showContours(contours,hierarchy,"CONTOURS IN D1",image)
                dt.showContoursAndCenters(circ_contours,hierarchy,circ_contour_centroids,"PASSING CIRCULAR CONTOURS IN D1 AND THEIR CENTROIDS",image)
                cv.imshow("THRESHED IMAGE OF ENCODING", encoded_src_threshed)
                cv.waitKey(0)

        if(not cid_found):
                return bit_encoding
        
        if(DECODER_SHOWCASE_MODE):
                show_image = image
                dt.showContoursAndCentersOnImage(circ_contours,hierarchy,circ_contour_centroids,"PASSING CIRCULAR CONTOURS IN D1 AND THEIR CENTROIDS",show_image); 
                cv.waitKey(0)
        
        decode_image = encoded_src_threshed
        decode_blur = cv.medianBlur( decode_image, 3 )
        _,decode_thresh_bin = cv.threshold( decode_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE )
        
        pre_bit_pass = True
        pre_bit_encoding = []
        segment_percentw_vec = []
        for ii in range(0,ENCODING_LENGTH):
                for jj in range(0,ENCODING_LENGTH):
                        if( ii*ENCODING_LENGTH + jj == cid_indx): 
                                pre_bit_encoding.append(-1) 
                                continue
                        segmentSubMatrix = decode_thresh_bin[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]/255
                        segment_percentw = cv.sumElems(segmentSubMatrix)[0]/segment_area
                        segment_percentw_vec.append(segment_percentw)

                        if( abs(segment_percentw - 0.5) > DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
                                if( (segment_percentw - 0.5) < 0 ):
                                        pre_bit_encoding.append(0)
                                else:
                                        pre_bit_encoding.append(1)
                        else:
                                pre_bit_encoding.append(-2)
                                pre_bit_pass = False


        if( not pre_bit_pass ):
                decode_image = encoded_src_threshed
                pre_bit_pass = True
                pre_bit_encoding = []
                segment_percentw_vec = []
                _,decode_image_tzthresh = cv.threshold( 255-decode_image, 255-DECODING_GREYSCALE_THRESH, 255, cv.THRESH_TOZERO)
                decode_image_tzthresh = 255 - decode_image_tzthresh
                decode_blur = cv.medianBlur( decode_image_tzthresh, 3 )
                _,decode_thresh_bin = cv.threshold( decode_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE )
                for ii in range(0,ENCODING_LENGTH):
                        for jj in range(0,ENCODING_LENGTH):
                                if( ii*ENCODING_LENGTH + jj == cid_indx): 
                                        pre_bit_encoding.append(-1) 
                                        continue
                                segmentSubMatrix = decode_thresh_bin[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]/255
                                segment_percentw = cv.sumElems(segmentSubMatrix)[0]/segment_area
                                segment_percentw_vec.append(segment_percentw)

                                if( abs(segment_percentw - 0.5) > DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
                                        if( (segment_percentw - 0.5) < 0 ):
                                                pre_bit_encoding.append(0)
                                        else:
                                                pre_bit_encoding.append(1)
                                else:
                                        pre_bit_encoding.append(-2)
                                        pre_bit_pass = False

        if( pre_bit_pass and cid_found):
                # APPARENTLY SWITCH/MATCH STATEMENTS ARE NEW TO PYTHON
                if( cid_indx == 0 ):
                        bit_encoding = [ int(pre_bit_encoding[1]),int(pre_bit_encoding[2]),int(pre_bit_encoding[3]),int(pre_bit_encoding[4]), \
                                         int(pre_bit_encoding[5]),int(pre_bit_encoding[6]),int(pre_bit_encoding[7]),int(pre_bit_encoding[8]) ]
                elif( cid_indx == 2 ):
                        bit_encoding = [ int(pre_bit_encoding[5]),int(pre_bit_encoding[8]),int(pre_bit_encoding[1]),int(pre_bit_encoding[4]), \
                                         int(pre_bit_encoding[7]),int(pre_bit_encoding[0]),int(pre_bit_encoding[3]),int(pre_bit_encoding[6]) ]
                elif( cid_indx == 6 ):
                        bit_encoding = [ int(pre_bit_encoding[3]),int(pre_bit_encoding[0]),int(pre_bit_encoding[7]),int(pre_bit_encoding[4]), \
                                         int(pre_bit_encoding[1]),int(pre_bit_encoding[8]),int(pre_bit_encoding[5]),int(pre_bit_encoding[2]) ]
                elif( cid_indx == 8 ):
                        bit_encoding = [ int(pre_bit_encoding[7]),int(pre_bit_encoding[6]),int(pre_bit_encoding[5]),int(pre_bit_encoding[4]), \
                                         int(pre_bit_encoding[3]),int(pre_bit_encoding[2]),int(pre_bit_encoding[1]),int(pre_bit_encoding[0]) ]
        # DETERMINE BIT ENCODING - END
        if(DECODER_SHOWCASE_MODE):
                show_image = image
                dt.showEncodingInformation( pre_bit_encoding, row_seg, ENCODING_LENGTH, "DECODED MESSAGE",  show_image ) 
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
        bitencoding = decodeImage(src,0)
        print(bitencoding)
        

if __name__ == '__main__':
        main()
