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

class utlift_classifier:

        cascade_debug_mode  = None
        cascade_on_hardware = None

        def __init__(self, cascade_on_hardware, cascade_debug_mode  = 3 ):
                self.cascade_on_hardware = cascade_on_hardware
                self.cascade_debug_mode = cascade_debug_mode
                pass

        def __del__(self):
                pass

        # TODO ADD CHECK FOR WHEN ID TOO MANY OR TOO LITTLE BARS WERE FOUND
        def extractD1Domain( self, image, debug_mode, on_hardware = False):

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
                unwarped_center = []
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
                
                # GET CENTER OF ENCODING (UNWARPED)
                try:
                        inv_transformation_mat = np.linalg.pinv( transformation_mat )
                        lx, ly, _ = warped_image.shape 
                        warped_center = np.array( [ [ [ lx/2, ly/2 ] ]], dtype="float32")
                        # print(warped_center)
                        unwarped_center = cv.perspectiveTransform(warped_center,inv_transformation_mat)
                        # print(unwarped_center)
                        # image_check = cv.circle(image.copy(), (int(unwarped_center[0][0][0]),int(unwarped_center[0][0][1])), 20, (int(255), int(0), int(0)), 3)
                        # cv.imwrite("check_1.png",image_check)
                except:
                        print("[ERROR]: UNWARPED CENTER POINT NOT DETERMINED... CONTINUING")
                        pass

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
                return encoded_eval_contours, cid_info, unwarped_center[0][0], rvec, tvec



        def determineEncodingFromD1Image( self, image, cid_info, debug_mode, on_hardware = False ):
        
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



        # def decodeImage( self, image ):
        #         DEBUG_D1E_FLAG = False
        #         DEBUG_DENC_FLAG = False
        #         if(self.cascade_debug_mode == 1):
        #                 DEBUG_D1E_FLAG = True
        #                 DEBUG_DENC_FLAG = False
        #         elif(self.cascade_debug_mode == 2):
        #                 DEBUG_D1E_FLAG = False
        #                 DEBUG_DENC_FLAG = True
        #         elif(self.cascade_debug_mode == 3):
        #                 DEBUG_D1E_FLAG = True
        #                 DEBUG_DENC_FLAG = True
        #         encodedImage, cid_info, crate_centerpt, _, _ = self.extractD1Domain(image, DEBUG_D1E_FLAG, self.cascade_on_hardware )
        #         if not cid_info.found:
        #                 return []
        #         if not len(encodedImage): 
        #                 print('\n[RESULT]: ENCODED ROI COULD NOT BE DETERMINED')
        #                 return []
        #         else:
        #                 bit_encoding = self.determineEncodingFromD1Image( encodedImage, cid_info, DEBUG_DENC_FLAG, self.cascade_on_hardware )
        #                 if not len(bit_encoding):
        #                         print('\n[RESULT]: MARKER ENCODING COULD NOT BE DECODED')
        #                 else:
        #                         print('\n[RESULT]: ' + str(bit_encoding))
                        
        #                 return bit_encoding, crate_centerpt



        def classifyImage( self, image, crnrs = None ):
                image_in = image
                if not ( crnrs is None ):
                        x1, y1, x2, y2 = int(crnrs[0]), int(crnrs[1]), int(crnrs[2]), int(crnrs[3])
                        image_in = image[y1:y2, x1:x2]

                DEBUG_D1E_FLAG = False
                DEBUG_DENC_FLAG = False
                if(self.cascade_debug_mode == 1):
                        DEBUG_D1E_FLAG = True
                        DEBUG_DENC_FLAG = False
                elif(self.cascade_debug_mode == 2):
                        DEBUG_D1E_FLAG = False
                        DEBUG_DENC_FLAG = True
                elif(self.cascade_debug_mode == 3):
                        DEBUG_D1E_FLAG = True
                        DEBUG_DENC_FLAG = True
                encodedImage, cid_info, crate_centerpt, _, _ = self.extractD1Domain(image_in, DEBUG_D1E_FLAG, self.cascade_on_hardware )
                if not cid_info.found:
                        return []
                if not len(encodedImage): 
                        print('\n[RESULT]: ENCODED ROI COULD NOT BE DETERMINED')
                        return []
                else:
                        bit_encoding = self.determineEncodingFromD1Image( encodedImage, cid_info, DEBUG_DENC_FLAG, self.cascade_on_hardware )
                        if not len(bit_encoding):
                                print('\n[RESULT]: MARKER ENCODING COULD NOT BE DECODED')
                        else:
                                print('\n[RESULT]: ' + str(bit_encoding))
                        
                        return bit_encoding, crate_centerpt

###############################################################################################################################################################################

class aruco_options:
        parameters  = None 
        dictionary  = None

        def __init__(self, parameters, dictionary ):
                self.parameters = parameters
                self.dictionary = dictionary

class aruco_classifier:
        options = None
        
        def __init__( self, options ):
                self.options = options

        def classifyImage( self, imgi  ):
                img                             = cv.cvtColor(imgi, cv.COLOR_BGR2GRAY)
                bbcrnrs_list, read_code_list, _ = cv.aruco.detectMarkers(img, self.options.dictionary, parameters=self.options.parameters)
                if len(bbcrnrs_list) > 0: 
                        cntr_pxlpt_list = []*len(bbcrnrs_list)
                        for ii in range( 0, len(bbcrnrs_list) ):
                                crnrs                   = bbcrnrs_list[ii].reshape((4, 2))
                                crnr_tl, _ , crnr_br, _ = crnrs
                                cntr_pxlpt              = ( (crnr_tl[0] + crnr_br[0])/2 , (crnr_tl[1] + crnr_br[1])/2 )
                                cntr_pxlpt_list[ii]     = cntr_pxlpt
                        return read_code_list, cntr_pxlpt_list
                return [],[]

        

# def main():
#         imageName = str( sys.argv[1] )

#         start = time.time()
#         src = cv.imread(imageName)
#         assert imageName is not None, "[ERROR] file could not be read, check with os.path.exists()"
#         bitencoding = decodeImage(src,dcdc.DECODER_DEBUG_MODE)

#         print("\nPREDICTED ENCODING:\n"+str(bitencoding))
#         print("Process Time: " + str( round( time.time() - start, 5 )  ) )

# if __name__ == '__main__':
#         main()
