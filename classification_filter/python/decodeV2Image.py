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
        rect_contours = []
        rect_contours_pass1 = []
        rect_contour_centroids = []
        rect_contour_angles = []
        rect_contour_areas = []
        poly_contour_areas = []
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
        bit_encoding = []
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
        cid_corner_indx = int(-1)
        cid_indx = int(-1) 
        cid_found = False
        crnr0_img, crnr1_img, crnr2_img, crnr3_img = [], [], [], []
        crnr0_cntrs, crnr1_cntrs, crnr2_cntrs, crnr3_cntrs  = [], [], [], []
        crnr0_total_area, crnr1_total_area, crnr2_total_area, crnr3_total_area = 0.0, 0.0, 0.0, 0.0
        crnr0_area_nonzero, crnr1_area_nonzero, crnr2_area_nonzero, crnr3_area_nonzero  = False, False, False, False 

        for ii in range(0,dcdc.ENCODING_LENGTH,2):
                for jj in range(0,dcdc.ENCODING_LENGTH,2):
                        indx = ii*dcdc.ENCODING_LENGTH + jj
                        cid_eval_SegementSubmatrix = src_eval_contours[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ]
                        if( indx == 0 ):
                                crnr0_img = cid_eval_SegementSubmatrix
                        elif( indx == 2 ):
                                crnr1_img = cid_eval_SegementSubmatrix
                        elif( indx == 6 ):
                                crnr2_img = cid_eval_SegementSubmatrix
                        elif( indx == 8 ):
                                crnr3_img = cid_eval_SegementSubmatrix

                        canny_output = cv.Canny( cid_eval_SegementSubmatrix, 10, 200 )
                        contours, hierarchy = cv.findContours( canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1 )
                        for kk in range(0,len(contours)):
                                approx_cp = cv.approxPolyDP( contours[kk], 0.01*cv.arcLength(contours[kk],True), True )
                                if( len(approx_cp) > dcdc.CIRC_IDENTIFIER_SIDES_THRESHOLD  ):
                                        approx_min_circ_center,approx_min_circ_radius = cv.minEnclosingCircle( contours[kk] )
                                        approx_cp_area = math.pi * pow(approx_min_circ_radius,2)
                                        if(    approx_cp_area < ( segment_area ) * (dcdc.SEGMENT_LCIRC_RATIO) * (1 + float(dcdc.CIRC_AREA_UPPER_PERCENT_THRESHOLD)/100) \
                                           and approx_cp_area > ( segment_area ) * (dcdc.SEGMENT_SCIRC_RATIO) * (1 - float(dcdc.CIRC_AREA_LOWER_PERCENT_THRESHOLD)/100)  ):
                                                if( indx == 0 ):
                                                        crnr0_cntrs.append( contours[kk] )
                                                        crnr0_total_area = crnr0_total_area + approx_cp_area
                                                        if( not crnr0_area_nonzero):
                                                                crnr0_area_nonzero = True
                                                elif( indx == 2 ):
                                                        crnr1_cntrs.append( contours[kk] )
                                                        crnr1_total_area = crnr1_total_area + approx_cp_area
                                                        if( not crnr1_area_nonzero):
                                                                crnr1_area_nonzero = True
                                                elif( indx == 6 ):
                                                        crnr2_cntrs.append( contours[kk] )
                                                        crnr2_total_area = crnr2_total_area + approx_cp_area
                                                        if( not crnr2_area_nonzero):
                                                                crnr2_area_nonzero = True
                                                elif( indx == 8 ):
                                                        crnr3_cntrs.append( contours[kk] )
                                                        crnr3_total_area = crnr3_total_area + approx_cp_area
                                                        if( not crnr3_area_nonzero):
                                                                crnr3_area_nonzero = True


        if( debug_mode ):
                print( "[DEBUG] CIRC-ID IS IN " + str(cid_indx) + "R-POSITION" )
                
                crnr0_show_img = crnr0_img
                crnr1_show_img = crnr1_img
                crnr2_show_img = crnr2_img
                crnr3_show_img = crnr3_img

                cv.imshow("THRESHED IMAGE OF ENCODING", src_eval_contours)
                dt.showContoursOnImage(crnr0_cntrs,hierarchy,"CORNER0 CONTOURS IN D1",crnr0_show_img)
                dt.showContoursOnImage(crnr1_cntrs,hierarchy,"CORNER1 CONTOURS IN D1",crnr1_show_img)
                dt.showContoursOnImage(crnr2_cntrs,hierarchy,"CORNER2 CONTOURS IN D1",crnr2_show_img)
                dt.showContoursOnImage(crnr3_cntrs,hierarchy,"CORNER3 CONTOURS IN D1",crnr3_show_img)
                cv.waitKey(0)

        if( not( crnr0_area_nonzero or crnr1_area_nonzero or crnr2_area_nonzero or crnr3_area_nonzero ) ):
                return bit_encoding
        else:
                cid_found = True
                cid_corner_indx = np.array( [ crnr0_total_area, crnr1_total_area, crnr2_total_area, crnr3_total_area ] ).argmax()
                
                if( cid_corner_indx == 0 ):
                        cid_indx = int(0)

                elif( cid_corner_indx == 1 ):
                        cid_indx = int(2)

                elif( cid_corner_indx == 2 ):
                        cid_indx = int(6)

                elif( cid_corner_indx == 3 ):
                        cid_indx = int(8)
        # LOCATING CIRCULAR IDENTIFIER - END

        
        if(dcdc.DECODER_SHOWCASE_MODE):
                show_image = image
                for ii in range(0,dcdc.ENCODING_LENGTH,2):
                        for jj in range(0,dcdc.ENCODING_LENGTH,2):
                                indx = ii*dcdc.ENCODING_LENGTH + jj
                                drawingSegementSubmatrix = show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ]
                                if( indx == 0 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr0_cntrs, hierarchy , drawingSegementSubmatrix )
                                elif( indx == 2 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr1_cntrs, hierarchy , drawingSegementSubmatrix )
                                elif( indx == 6 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr2_cntrs, hierarchy , drawingSegementSubmatrix )
                                elif( indx == 8 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr3_cntrs, hierarchy , drawingSegementSubmatrix )
                cv.imshow("PASSING CIRCULAR CONTOURS IN D1 AND THEIR CENTROIDS",show_image)
                cv.waitKey(0)
        

        # DETERMINE BIT ENCODING - START
        decode_image = src_eval_contours
        decode_blur = cv.medianBlur( decode_image, 3 )

        pre_bit_pass = True
        pre_bit_encoding = []
        segment_percentw_vec = []
        row_sub_seg = int(row_seg/2)
        col_sub_seg = int(col_seg/2)
        indx0_img, indx1_img, indx2_img, indx3_img, indx4_img, indx5_img, indx6_img, indx7_img, indx8_img  = [], [], [], [], [], [], [], [], []

        for ii in range(0,dcdc.ENCODING_LENGTH):
                for jj in range(0,dcdc.ENCODING_LENGTH):

                        indx = ii*dcdc.ENCODING_LENGTH + jj

                        if( indx == cid_indx): 
                                pre_bit_encoding.append(-1) 
                                continue

                        segmentSubMatrix = decode_blur[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]
                        _,sSM_thresh_bin = cv.threshold( segmentSubMatrix.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU )

                        if( indx == 0 ):
                                indx0_img = sSM_thresh_bin
                        elif( indx == 1 ):
                                indx1_img = sSM_thresh_bin
                        elif( indx == 2 ):
                                indx2_img = sSM_thresh_bin
                        elif( indx == 3 ):
                                indx3_img = sSM_thresh_bin
                        elif( indx == 4 ):
                                indx4_img = sSM_thresh_bin
                        elif( indx == 5 ):
                                indx5_img = sSM_thresh_bin
                        elif( indx == 6 ):
                                indx6_img = sSM_thresh_bin
                        elif( indx == 7 ):
                                indx7_img = sSM_thresh_bin
                        elif( indx == 8 ):
                                indx8_img = sSM_thresh_bin

                        sSM_thresh_bin = sSM_thresh_bin/255

                        segment_percentw = 0.0
                        for kk in range(0,2):
                                for ll in range(0,2):
                                        if( kk*2 + ll != cid_corner_indx):
                                                segmentSubSubMatrix = sSM_thresh_bin[ kk*row_sub_seg:(kk+1)*row_sub_seg-1, ll*col_sub_seg:(ll+1)*col_sub_seg-1 ]
                                                segment_percentw = segment_percentw + cv.sumElems(segmentSubSubMatrix)[0]/(0.75*segment_area)
                        
                        segment_percentw_vec.append(segment_percentw)

                        if( abs(segment_percentw - 0.5) > dcdc.DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
                                if( (segment_percentw - 0.5) < 0 ):
                                        pre_bit_encoding.append(0)
                                else:
                                        pre_bit_encoding.append(1)
                        else:
                                pre_bit_encoding.append(-2)
                                pre_bit_pass = False


        if(debug_mode):
                show_image = src_gray
                for ii in range(0,dcdc.ENCODING_LENGTH):
                        for jj in range(0,dcdc.ENCODING_LENGTH):
                                indx = ii*dcdc.ENCODING_LENGTH + jj
                                if( indx == cid_indx): 
                                        continue
                                elif( indx == 0 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx0_img
                                elif( indx == 1 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx1_img
                                elif( indx == 2 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx2_img
                                elif( indx == 3 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx3_img
                                elif( indx == 4 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx4_img
                                elif( indx == 5 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx5_img
                                elif( indx == 6 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx6_img
                                elif( indx == 7 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx7_img
                                elif( indx == 8 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx8_img

                cv.imshow("BINARIZED D1 SEGMENTS",show_image)
                cv.waitKey(0)


        if( not pre_bit_pass ):
                decode_image = encoded_src_threshed
                pre_bit_pass = True
                pre_bit_encoding = []
                segment_percentw_vec = []
                _,decode_image_tzthresh = cv.threshold( 255-decode_image, 255-dcdc.DECODING_GREYSCALE_THRESH, 255, cv.THRESH_TOZERO)
                decode_image_tzthresh = 255 - decode_image_tzthresh
                decode_blur = cv.medianBlur( decode_image_tzthresh, 3 )
                for ii in range(0,dcdc.ENCODING_LENGTH):
                        for jj in range(0,dcdc.ENCODING_LENGTH):
                                if( ii*dcdc.ENCODING_LENGTH + jj == cid_indx): 
                                        pre_bit_encoding.append(-1) 
                                        continue
                                segmentSubMatrix = decode_blur[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]/255
                                _,sSM_thresh_bin = cv.threshold( segmentSubMatrix.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE )
                                sSM_thresh_bin = sSM_thresh_bin/255

                                segment_percentw = 0.0
                                for kk in range(0,2):
                                        for ll in range(0,2):
                                                if( kk*2 + ll != cid_corner_indx):
                                                        segmentSubSubMatrix = sSM_thresh_bin[ kk*row_sub_seg:(kk+1)*row_sub_seg-1, ll*col_sub_seg:(ll+1)*col_sub_seg-1 ]
                                                        segment_percentw = segment_percentw + cv.sumElems(segmentSubSubMatrix)[0]/(0.75*segment_area)
                                
                                segment_percentw_vec.append(segment_percentw)

                                if( abs(segment_percentw - 0.5) > dcdc.DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
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
