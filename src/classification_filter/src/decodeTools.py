import cv2 as cv
import math
import numpy as np
import metrics2D as est
import decodeConst as dcdc
import drawingTools as dt

def determineCandidateRectIDbars( src_atleast_grys, debug_mode, on_hardware = False):
        img_height, img_width = src_atleast_grys.shape
        rect_contours, rect_contours_pass1, rect_contour_centroids, rect_contour_angles, rect_contour_areas, poly_contour_areas = [], [], [], [], [], []
        canny_output = cv.Canny( src_atleast_grys, 10, 200, True )
        _, contours, _ = cv.findContours( canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS )
        for ii in range(0,len(contours)):
                approx_cp = cv.approxPolyDP( contours[ii], 0.025*cv.arcLength(contours[ii],True), True )
                approx_cp_area = abs( cv.contourArea(approx_cp) )
                if( (     approx_cp_area > img_height * img_width * pow(float(dcdc.RECT_AREA_PERCENT_THRESHOLD)/100,2) ) \
                      and len(approx_cp) >= dcdc.RECT_IDENTIFIER_SIDES_LOWER_THRESHOLD \
                      and len(approx_cp) <= dcdc.RECT_IDENTIFIER_SIDES_UPPER_THRESHOLD  ):
                        rect_contours_pass1.append( contours[ii] )
                        approx_rect_center, approx_rect_size, approx_rect_angle  = cv.minAreaRect(approx_cp)
                        approx_aspectr = max( approx_rect_size[0]/approx_rect_size[1], approx_rect_size[1]/approx_rect_size[0] )

                        if(    approx_aspectr > (1 - float(dcdc.RECT_ASPECT_RATIO_LOWER_PERECNT_ERROR_THRESHOLD)/100) * dcdc.MEASURED_ASPECT_RATIO 
                           and approx_aspectr < (1 + float(dcdc.RECT_ASPECT_RATIO_UPPER_PERECNT_ERROR_THRESHOLD)/100) * dcdc.MEASURED_ASPECT_RATIO ):
                                rect_contour_areas.append( approx_rect_size[0]*approx_rect_size[1] )
                                poly_contour_areas.append( approx_cp_area ) # NOT THE SAME AS THE ONE ABOVE
                                rect_contours.append( contours[ii] )
                                rect_contour_centroids.append(approx_rect_center)
                                if( approx_rect_size[0] > approx_rect_size[1] ):
                                        rect_contour_angles.append(approx_rect_angle)
                                else:
                                        rect_contour_angles.append( 90. + approx_rect_angle )
        
        if(debug_mode and (not on_hardware) ):
                dt.showContours(contours,"CONTOURS IN D3",src_atleast_grys.shape)
                dt.showContours(rect_contours_pass1,"CONTOURS THAT PASS EVERYTHING BUT ASPECT-RATIO",src_atleast_grys.shape)
                cv.waitKey(0)

        return rect_contours, rect_contour_centroids, rect_contour_angles, rect_contour_areas, poly_contour_areas


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
        
        return warped_image,fc_median_distance, transformationMat


def determineWarpedImageFrom2or3IdBars(image, rect_contour_centroids, rect_contour_angles, DEBUG_MODE):
        # TODO: DETERMINE THE ROATION ANGLE IS INCONSITENT (OFF BY 180 IN CERTAIN UNIDENTIFIED CONDITIONS)
        ref_angle = rect_contour_angles[0]
        slope = math.tan ( ref_angle * math.pi / 180.0 )
        yinter = rect_contour_centroids[0][1] - slope*rect_contour_centroids[0][0]
        above_reference_line = ( rect_contour_centroids[1][1] - (slope*rect_contour_centroids[1][0] + yinter) ) > 0

        fc_median_distance = 0.
        if( abs(ref_angle - rect_contour_angles[0]) < 45.0 ):
                fc_median_distance = 2 * est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1]) / math.sqrt(2)
        else:
                fc_median_distance = est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1])
        
        delta_angle = 0

        # TODO: TEST THIS DELTA ANGLE DETERMINATION
        if(above_reference_line):
                delta_angle = -ref_angle + 90
        else:
                delta_angle = -ref_angle - 90

        img_height, img_width,_ = image.shape
        rotatedimageSize = ( int(2*img_height) , int(2*img_width)  )

        transformationMat = cv.getRotationMatrix2D( rect_contour_centroids[0], delta_angle, 1 )
        rotatedImage = cv.warpAffine(image, transformationMat, rotatedimageSize )
        warped_image = rotatedImage[ int(rect_contour_centroids[0][1]-0.5*fc_median_distance):int( rect_contour_centroids[0][1]+0.5*fc_median_distance ) , \
                                     int(rect_contour_centroids[0][0]):int(rect_contour_centroids[0][0]+fc_median_distance ) ]
        
        return warped_image, fc_median_distance


def attemptIdBarCorrections( image_shape, rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas, debug_mode, on_hardware = False  ):
        # 1ST PROTECTION FOR MORE THAN 4 BARS (CENTROID PROXIMITY) -START
        rect_contours_corr, rect_contour_centroids_corr, rect_contour_angles_corr, rect_contour_areas_corr, poly_contour_areas_corr = [], [], [], [], []
        rect_contours_corr.append( rect_contours[0] )
        rect_contour_centroids_corr.append( rect_contour_centroids[0] )
        rect_contour_angles_corr.append( rect_contour_angles[0] )
        rect_contour_areas_corr.append( rect_contour_areas[0] )
        poly_contour_areas_corr.append( poly_contour_areas[0] )

        for ii in range( 1, len(rect_contour_centroids) ):
                pass_flag = True
                
                for jj in range( 0, len(rect_contour_centroids_corr) ):
                        if( est.euclideanDistance( rect_contour_centroids[ii], rect_contour_centroids_corr[jj] ) \
                             <= (dcdc.CONTOUR_EDIST_PERCENT_THRES/100) * ( image_shape[0] + image_shape[1] )/2 ):
                                # DO A SIZE CHECK TO CHOOSE WHICH TO REPLACE 
                                if( rect_contour_areas[ii] > rect_contour_areas_corr[jj] ):
                                        rect_contours_corr[jj] = rect_contours[ii] 
                                        rect_contour_centroids_corr[jj] = rect_contour_centroids[ii] 
                                        rect_contour_angles_corr[jj] = rect_contour_angles[ii] 
                                        rect_contour_areas_corr[jj] = rect_contour_areas[ii]  
                                        poly_contour_areas_corr[jj] = poly_contour_areas[ii] 

                                pass_flag = False 
                                break
                
                if( pass_flag ):
                        rect_contours_corr.append( rect_contours[ii] )
                        rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
                        rect_contour_angles_corr.append( rect_contour_angles[ii] )
                        rect_contour_areas_corr.append( rect_contour_areas[ii]  )
                        poly_contour_areas_corr.append( poly_contour_areas[ii] )

        
        rect_contours = rect_contours_corr
        rect_contour_centroids = rect_contour_centroids_corr
        rect_contour_angles = rect_contour_angles_corr
        rect_contour_areas = rect_contour_areas_corr
        poly_contour_areas = poly_contour_areas_corr
        if(debug_mode and (not on_hardware)):
                        dt.showContoursAndAreas(rect_contours,"PASSED FIRST PROTECTION",image_shape)
                        cv.waitKey(0)
        # 1ST PROTECTION FOR MORE THAN 4 BARS (CENTROID PROXIMITY) - END

############################################################################## ELIMINATE SMALL CONTOURS - START

        rect_contours_corr, rect_contour_centroids_corr, rect_contour_angles_corr, rect_contour_areas_corr, poly_contour_areas_corr = [], [], [], [], []

        for ii in range( 0, len(rect_contours) ):

                if rect_contour_areas[ii] > ((image_shape[0] * image_shape[1]) * float((dcdc.RECT_MINIMUM_SCREEN_PERCENTAGE/100))):
                        rect_contours_corr.append( rect_contours[ii] )
                        rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
                        rect_contour_angles_corr.append( rect_contour_angles[ii] )
                        rect_contour_areas_corr.append( rect_contour_areas[ii]  )
                        poly_contour_areas_corr.append( poly_contour_areas[ii] )

        rect_contours = rect_contours_corr
        rect_contour_centroids = rect_contour_centroids_corr
        rect_contour_angles = rect_contour_angles_corr
        rect_contour_areas = rect_contour_areas_corr
        poly_contour_areas = poly_contour_areas_corr

        if(debug_mode and (not on_hardware) ):
                dt.showContoursAndAreas(rect_contours,"PASSED SECOND PROTECTION",image_shape)
                cv.waitKey(0)

############################################################################## BOUNDING BOX AREA/CONTOUR AREA RATIO TEST - START

        rect_contours_corr, rect_contour_centroids_corr, rect_contour_angles_corr, rect_contour_areas_corr, poly_contour_areas_corr = [], [], [], [], []
        rect_contours_recovery, rect_contour_centroids_recovery, rect_contour_angles_recovery, rect_contour_areas_recovery, poly_contour_areas_recovery = [], [], [], [], []
        for ii in range(0,len(rect_contours)):
                contour_area_ratio = poly_contour_areas[ii]/rect_contour_areas[ii] ##IDEALLY = 1
                if ((contour_area_ratio < 1 + (float(dcdc.RECT_BOUNDING_CONTOUR_AREA_THRESHOLD)/100)) 
                and (contour_area_ratio > 1 - (float(dcdc.RECT_BOUNDING_CONTOUR_AREA_THRESHOLD)/100))):
                        rect_contours_corr.append( rect_contours[ii] )
                        rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
                        rect_contour_angles_corr.append( rect_contour_angles[ii] )
                        rect_contour_areas_corr.append( rect_contour_areas[ii]  )
                        poly_contour_areas_corr.append( poly_contour_areas[ii] )
                else:
                        rect_contours_recovery.append( rect_contours[ii] )
                        rect_contour_centroids_recovery.append( rect_contour_centroids[ii] )
                        rect_contour_angles_recovery.append( rect_contour_angles[ii] )
                        rect_contour_areas_recovery.append( rect_contour_areas[ii]  )
                        poly_contour_areas_recovery.append( poly_contour_areas[ii] )   

        rect_contours = rect_contours_corr
        rect_contour_centroids = rect_contour_centroids_corr
        rect_contour_angles = rect_contour_angles_corr
        rect_contour_areas = rect_contour_areas_corr
        poly_contour_areas = poly_contour_areas_corr

        if debug_mode:
                if len(rect_contours) == 4 or len(rect_contours) == 0:
                        print("\n[DEBUG]: %2d MARKERS DETECTED AT SECOND PROTECTION. EXITING BAR CORRECTIONS" %(len(rect_contours)))
                if (not on_hardware):
                        dt.showContoursAndAreas(rect_contours,"PASSED THIRD PROTECTION",image_shape)
                        cv.waitKey(0)
                

        if len(rect_contours) == 4 or len(rect_contours_recovery) == 0 or len(rect_contours) == 0:
                return rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas


        # BOUNDING BOX AREA/CONTOUR AREA RATIO TEST  - END

############################################################################## CONTOUR RECOVERY - START

        ###################################################################### RELEVANT BOUNDING BOX AREA  - START

        rect_contours_recovery2, rect_contour_centroids_recovery2, rect_contour_angles_recovery2, rect_contour_areas_recovery2, poly_contour_areas_recovery2 = [], [], [], [], []

        #SET UPPER AND LOWER THREHSOLDS
        poly_lowest_perim = cv.arcLength(rect_contours[0],True)
        # poly_highest_perim = cv.arcLength(rect_contours[0],True)
        rect_lowest_area = rect_contour_areas[0]
        rect_highest_area = rect_contour_areas[0]

        for ii in range(0,len(rect_contours)):
                if(cv.arcLength(rect_contours[ii],True) < poly_lowest_perim):
                        poly_lowest_perim = cv.arcLength(rect_contours[ii],True)
                # else:
                #         poly_highest_perim = cv.arcLength(rect_contours[ii],True)
                if(rect_contour_areas[ii] < rect_lowest_area):
                        rect_lowest_area = rect_contour_areas[ii]
                else:
                        rect_highest_area = rect_contour_areas[ii]

        for ii in range(0,len(rect_contours_recovery)):
                if (rect_contour_areas_recovery[ii] <= rect_highest_area * (1 + float((dcdc.RECT_UPPER_COMPARISON_PERCENT_ERROR_THRESHOLD)/100)) 
                and rect_contour_areas_recovery[ii] >= rect_lowest_area * (1 - float((dcdc.RECT_LOWER_COMPARISON_PERCENT_ERROR_THRESHOLD)/100))):
                        rect_contours_recovery2.append( rect_contours_recovery[ii] )
                        rect_contour_centroids_recovery2.append( rect_contour_centroids_recovery[ii] )
                        rect_contour_angles_recovery2.append( rect_contour_angles_recovery[ii] )
                        rect_contour_areas_recovery2.append( rect_contour_areas_recovery[ii]  )
                        poly_contour_areas_recovery2.append( poly_contour_areas_recovery[ii] ) 

        if(debug_mode and (not on_hardware)):
                dt.showContoursAndAreas(rect_contours_recovery2,"PASSED FIRST RECOVERY TEST",image_shape)
                cv.waitKey(0)
                
        # RELE BOUNDARY BOX AREA COMPARISON - END

        ######################################################################  RELEVANT CONTOUR PERIMETER - START

        rect_contours_recovery3, rect_contour_centroids_recovery3, rect_contour_angles_recovery3, rect_contour_areas_recovery3, poly_contour_areas_recovery3 = [], [], [], [], []

        if len(rect_contours_recovery2) > 0:

                # rect_perim_avg = 0
                rect_perim_total = 0
                for ii in range(0,len(rect_contours)):
                        rect_perim_total = rect_perim_total + cv.arcLength(rect_contours[ii],True)
                # rect_perim_avg = rect_perim_total / len(rect_contours)

                for ii in range(0,len(rect_contours_recovery2)):
                        poly_perim_fail2 = cv.arcLength(rect_contours_recovery2[ii],True)
                        # RECTANGLES ARE OFTEN OVERESTIMATED SO WE ONLY SET LOWER BOUND
                        if ( poly_perim_fail2 >= poly_lowest_perim * (1 - float((dcdc.RECT_LOWER_COMPARISON_PERCENT_ERROR_THRESHOLD)/100))):
                                rect_contours_recovery3.append( rect_contours_recovery2[ii] )
                                rect_contour_centroids_recovery3.append( rect_contour_centroids_recovery2[ii] )
                                rect_contour_angles_recovery3.append( rect_contour_angles_recovery2[ii] )
                                rect_contour_areas_recovery3.append( rect_contour_areas_recovery2[ii]  )
                                poly_contour_areas_recovery3.append( poly_contour_areas_recovery2[ii] ) 
                
                if(debug_mode and (not on_hardware)):
                        dt.showContoursAndPerimeters(rect_contours_recovery3,"PASSED SECOND RECOVERY TEST",image_shape)
                        cv.waitKey(0)
                


        # CONTOUR PERIMETER COMPARISON - END

        ###################################################################### DISTANCE FROM MARKER CENTROID TEST - START
        #TODO ASPECT RATIO FOR ANGLES TO ENSURE IT ISNT FLIPPED
        if len(rect_contours_recovery3) > 0:
                marker_centroid = findMarkerCentroid(rect_contour_centroids, rect_contour_angles, debug_mode)
                if len(rect_contours) == 3:
                        avg_dist_from_centroid, dist_total = 0,0
                        #FIND AVG DISTANCE TO MARKER CENTROID
                        for ii in range(0,len(rect_contours)):
                                dist_total = dist_total + est.euclideanDistance(rect_contour_centroids[ii] , marker_centroid)
                        avg_dist_from_centroid_avg = dist_total / len(rect_contours)
                        #IF DISTANCE OF RECOVERY CONTOUR IS TOO LARGE THEN ELIMINATE IT
                        for ii in range(0,len(rect_contours_recovery3)):
                                if est.euclideanDistance(rect_contour_centroids_recovery3[ii],marker_centroid) <  avg_dist_from_centroid_avg * (1 + float(dcdc.RECT_CENTROID_DISTANCE_THRESHOLD/100)):
                                        rect_contours.append( rect_contours_recovery3[ii] )
                                        rect_contour_centroids.append( rect_contour_centroids_recovery3[ii] )
                                        rect_contour_angles.append( rect_contour_angles_recovery3[ii] )
                                        rect_contour_areas.append( rect_contour_areas_recovery3[ii]  )
                                        poly_contour_areas.append( poly_contour_areas_recovery3[ii] ) 
                if len(rect_contours) == 2:   
                                #DISTANCE TEST FROM MARKER CENTER
                                avg_dist_from_centroid = (est.euclideanDistance(rect_contour_centroids[0],marker_centroid) + est.euclideanDistance(rect_contour_centroids[1],marker_centroid)) / 2
                                for ii in range(0,len(rect_contours_recovery3)):
                                        if est.euclideanDistance(rect_contour_centroids_recovery3[ii],marker_centroid) <=  (avg_dist_from_centroid * (1 + float(dcdc.RECT_CENTROID_DISTANCE_THRESHOLD/100))):
                                                rect_contours.append( rect_contours_recovery3[ii] )
                                                rect_contour_centroids.append( rect_contour_centroids_recovery3[ii] )
                                                rect_contour_angles.append( rect_contour_angles_recovery3[ii] )
                                                rect_contour_areas.append( rect_contour_areas_recovery3[ii]  )
                                                poly_contour_areas.append( poly_contour_areas_recovery3[ii] )
         
        if(debug_mode and (not on_hardware)):
                dt.showContours(rect_contours,"AFTER RECOVERY",image_shape)
                cv.waitKey(0)

        # CENTROID GEOMETRY TEST - END


        # FINAL PROTECTION FOR MORE THAN 4 BARS ( RECT APPROX STD METHOD ) - START
        if( len(rect_contour_centroids) > 1 ):

                rect_contours_corr, rect_contour_centroids_corr, rect_contour_angles_corr, rect_contour_areas_corr, poly_contour_areas_corr = [], [], [], [], []
                for ii in range( 0, len(rect_contour_areas) ):
                        pass_flag = True
                        for jj in range( 0, len(rect_contour_areas) ):
                                relative_size_perc = rect_contour_areas[ii]/rect_contour_areas[jj]
                                if( relative_size_perc < (dcdc.REL_RECT_SIZE_PERCENT_THRESH/100) ):
                                        pass_flag = False
                                        break

                        if pass_flag:
                                rect_contours_corr.append( rect_contours[ii] )
                                rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
                                rect_contour_angles_corr.append( rect_contour_angles[ii] )
                                rect_contour_areas_corr.append( rect_contour_areas[ii]  )
                                poly_contour_areas_corr.append( poly_contour_areas[ii] )
                
                rect_contours = rect_contours_corr
                rect_contour_centroids = rect_contour_centroids_corr
                rect_contour_angles = rect_contour_angles_corr
                rect_contour_areas = rect_contour_areas_corr
                poly_contour_areas = poly_contour_areas_corr

                # CUTOFF IF TOO MANY 
                if( len(rect_contour_areas) > dcdc.RECT_CUTOFF_SIZE ):
                        sorted_indx = np.argsort(  np.array( rect_contour_areas ) )
                        # TRIM THE NUMBER OF CONTOURS BASED ON AREA
                        rect_contours = [ rect_contours[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        rect_contour_centroids = [ rect_contour_centroids[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        rect_contour_angles = [ rect_contour_angles[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        rect_contour_areas = [ rect_contour_areas[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        poly_contour_areas = [ poly_contour_areas[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]

                if(debug_mode and (not on_hardware)):
                        dt.showContours(rect_contours,"CONTOURS AFTER TRIMMING",image_shape)
                        cv.waitKey(0)

                size_before = len(rect_contour_areas)
                while( len(rect_contour_areas) > 4 ):
                        rect_contours_corr, rect_contour_centroids_corr, rect_contour_angles_corr, rect_contour_areas_corr, poly_contour_areas_corr = [], [], [], [], []
                        rect_contour_areas_mean = sum(  rect_contour_areas )/len(rect_contour_areas)
                        var = sum(  [ pow( recti_area - rect_contour_areas_mean, 2 ) for recti_area in rect_contour_areas]  ) / len(rect_contour_areas)
                        std = pow( var, 0.5 )
                        for ii in range( 0, len(rect_contour_centroids) ):
                                if(  abs(rect_contour_areas_mean-rect_contour_areas[ii]) <= std ):
                                        rect_contours_corr.append( rect_contours[ii] )
                                        rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
                                        rect_contour_angles_corr.append( rect_contour_angles[ii] )
                                        rect_contour_areas_corr.append( rect_contour_areas[ii]  )
                                        poly_contour_areas_corr.append( poly_contour_areas[ii] )

                        rect_contours = rect_contours_corr
                        rect_contour_centroids = rect_contour_centroids_corr
                        rect_contour_angles = rect_contour_angles_corr
                        rect_contour_areas = rect_contour_areas_corr
                        poly_contour_areas = poly_contour_areas_corr

                        if( len(rect_contour_areas) ) == size_before:
                                break 
                        else:
                                size_before = len(rect_contour_areas) 

                if(debug_mode and (not on_hardware) ):
                        dt.showContours(rect_contours,"CONTOURS AFTER FINAL PROTECTION",image_shape)
                        cv.waitKey(0)
        # 2ND PROTECTION FOR MORE THAN 4 BARS ( RECT APPROX STD METHOD ) - END


        # # 3RD PROTECTION FOR MORE THAN 4 BARS ( CONTOUR TRUE STD METHOD ) - START
        # if( len(rect_contour_centroids) > 1 ):
        #         rect_contours_corr = []
        #         rect_contour_centroids_corr = []
        #         rect_contour_angles_corr = []
        #         rect_contour_areas_corr = []
        #         poly_contour_areas_corr = []
                
        #         # CUTOFF IF TOO MANY 
        #         if( len(rect_contour_areas) > dcdc.RECT_CUTOFF_SIZE ):
        #                 sorted_indx = np.argsort(  np.array( rect_contour_areas_corr ) )
        #                 # TRIM THE NUMBER OF CONTOURS BASED ON AREA
        #                 rect_contours = [ rect_contours[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
        #                 rect_contour_centroids = [ rect_contour_centroids[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
        #                 rect_contour_angles = [ rect_contour_angles[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
        #                 rect_contour_areas = [ rect_contour_areas[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
        #                 poly_contour_areas = [ poly_contour_areas[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]


        #         poly_contour_areas_mean = sum(  poly_contour_areas )/len(poly_contour_areas)
        #         var = sum(  [ pow( polyi_area - poly_contour_areas_mean, 2 ) for polyi_area in poly_contour_areas]  ) / len(poly_contour_areas)
        #         std = pow( var, 0.5 )
        #         for ii in range( 0, len(rect_contour_centroids) ):
        #                 if(  abs(poly_contour_areas_mean-poly_contour_areas[ii]) <= std ):
        #                         rect_contours_corr.append( rect_contours[ii] )
        #                         rect_contour_centroids_corr.append( rect_contour_centroids[ii] )
        #                         rect_contour_angles_corr.append( rect_contour_angles[ii] )
        #                         rect_contour_areas_corr.append( rect_contour_areas[ii]  )
        #                         poly_contour_areas_corr.append( poly_contour_areas[ii] )

        #         rect_contours = rect_contours_corr
        #         rect_contour_centroids = rect_contour_centroids_corr
        #         rect_contour_angles = rect_contour_angles_corr
        #         rect_contour_areas = rect_contour_areas_corr
        #         poly_contour_areas = poly_contour_areas_corr
        #         if(debug_mode):
        #                 dt.showContours(rect_contours,"CONTOURS AFTER THIRD PROTECTION",image_shape)
        #                 cv.waitKey(0)
        # # 3RD PROTECTION FOR MORE THAN 4 BARS ( CONTOUR TRUE STD METHOD ) - END

        return rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas


def determineCIDIndices( src_eval_contours, row_seg, col_seg, segment_area, debug_mode, on_hardware = False ):
        # LOCATING CIRCULAR IDENTIFIER - START
        cid_corner_indx = int(-1)
        cid_indx = int(-1) 
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
                        _, contours, _ = cv.findContours( canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1 )
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
                print( "\n[DEBUG]: CIRC-ID IS IN " + str(cid_indx) + "R-POSITION" )
                if(not on_hardware):
                        crnr0_show_img = crnr0_img
                        crnr1_show_img = crnr1_img
                        crnr2_show_img = crnr2_img
                        crnr3_show_img = crnr3_img

                        cv.imshow("THRESHED IMAGE OF ENCODING", src_eval_contours)
                        dt.showContoursOnImage(crnr0_cntrs,"CORNER0 CONTOURS IN D1",crnr0_show_img)
                        dt.showContoursOnImage(crnr1_cntrs,"CORNER1 CONTOURS IN D1",crnr1_show_img)
                        dt.showContoursOnImage(crnr2_cntrs,"CORNER2 CONTOURS IN D1",crnr2_show_img)
                        dt.showContoursOnImage(crnr3_cntrs,"CORNER3 CONTOURS IN D1",crnr3_show_img)
                        cv.waitKey(0)

        if(dcdc.DECODER_SHOWCASE_MODE and (not on_hardware)):
                show_image = src_eval_contours.copy()
                for ii in range(0,dcdc.ENCODING_LENGTH,2):
                        for jj in range(0,dcdc.ENCODING_LENGTH,2):
                                indx = ii*dcdc.ENCODING_LENGTH + jj
                                drawingSegementSubmatrix = show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ]
                                if( indx == 0 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr0_cntrs , drawingSegementSubmatrix )
                                elif( indx == 2 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr1_cntrs , drawingSegementSubmatrix )
                                elif( indx == 6 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr2_cntrs , drawingSegementSubmatrix )
                                elif( indx == 8 ):
                                        show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = \
                                                dt.drawContoursOnImage( crnr3_cntrs , drawingSegementSubmatrix )
                cv.imshow("PASSING CIRCULAR CONTOURS IN D1 AND THEIR CENTROIDS",show_image)
                cv.waitKey(0)

        if( not( crnr0_area_nonzero or crnr1_area_nonzero or crnr2_area_nonzero or crnr3_area_nonzero ) ):
                return cid_corner_indx, cid_indx, False
        
        cid_corner_indx = np.array( [ crnr0_total_area, crnr1_total_area, crnr2_total_area, crnr3_total_area ] ).argmax()        
        if( cid_corner_indx == 0 ):
                cid_indx = int(0)
        elif( cid_corner_indx == 1 ):
                cid_indx = int(2)
        elif( cid_corner_indx == 2 ):
                cid_indx = int(6)
        elif( cid_corner_indx == 3 ):
                cid_indx = int(8)

        return cid_corner_indx, cid_indx, True
        # LOCATING CIRCULAR IDENTIFIER - END


def evaluateV2BitEncoding( src_atleast_grys, row_seg, col_seg, segment_area, cid_indx, cid_corner_indx, debug_mode, on_hardware = False ):
        decode_image = src_atleast_grys
        decode_blur = cv.medianBlur( decode_image, 3 )
        row_sub_seg = int(row_seg/2)
        col_sub_seg = int(col_seg/2)
        sub_segment_area = row_seg * col_seg/4
        indx0_img, indx1_img, indx2_img, indx3_img, indx4_img, indx5_img, indx6_img, indx7_img, indx8_img  = [], [], [], [], [], [], [], [], []

        pre_bit_pass = True
        enc_bit_encoding = []
        id_bit_encoding = []
        segment_percentw_vec = []
        segment_id_percentw_vec = []

        for ii in range(0,dcdc.ENCODING_LENGTH):
                for jj in range(0,dcdc.ENCODING_LENGTH):

                        indx = ii*dcdc.ENCODING_LENGTH + jj

                        if( indx == cid_indx): 
                                enc_bit_encoding.append(-1) 
                                id_bit_encoding.append(-1) 
                                continue

                        segmentSubMatrix = decode_blur[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]
                        _,sSM_thresh_bin = cv.threshold( segmentSubMatrix.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU )
                        
                        if(debug_mode):
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
                        segment_id_percentw = 0.0
                        for kk in range(0,2):
                                for ll in range(0,2):
                                        segmentSubSubMatrix = sSM_thresh_bin[ kk*row_sub_seg:(kk+1)*row_sub_seg-1, ll*col_sub_seg:(ll+1)*col_sub_seg-1 ]
                                        if( kk*2 + ll != cid_corner_indx):
                                                segment_percentw = segment_percentw + cv.sumElems(segmentSubSubMatrix)[0]/( 3*sub_segment_area )
                                        else:
                                                segment_id_percentw = segment_id_percentw + cv.sumElems(segmentSubSubMatrix)[0]/( sub_segment_area )
                        
                        segment_percentw_vec.append( segment_percentw )
                        segment_id_percentw_vec.append( segment_id_percentw )


                        if( abs(segment_percentw - 0.5) > dcdc.DECODING_ENC_CONFIDENCE_THRESHOLD - 0.5  ):
                                if( (segment_percentw - 0.5) < 0 ):
                                        enc_bit_encoding.append(0)
                                else:
                                        enc_bit_encoding.append(1)
                        else:
                                enc_bit_encoding.append(-2)
                                pre_bit_pass = False


                        if( abs(segment_id_percentw - 0.5) > dcdc.DECODING_ID_CONFIDENCE_THRESHOLD - 0.5  ):
                                if( (segment_id_percentw - 0.5) < 0 ):
                                        id_bit_encoding.append(0)
                                else:
                                        id_bit_encoding.append(1)
                        else:
                                id_bit_encoding.append(-2)
                                pre_bit_pass = False

        if(debug_mode):
                #print('[DEBUG]: ENCODING INFORMATION FOR 1ST TRIAL:')
                print('\n[DEBUG]: ENCODED SEGMENT: ' + str(enc_bit_encoding) )
                print('\n[DEBUG]: IDENTIFIER SEGMENT: ' + str(id_bit_encoding) )
                print('\n[DEBUG]: ENCODED SEGMENT PW: ' + str(segment_percentw_vec) )
                print('\n[DEBUG]: IDENTIFIER SEGMENT PW: ' + str(segment_id_percentw_vec) )


        if(debug_mode and (not on_hardware)):
                show_image = src_atleast_grys.copy()
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

                cv.imshow("TRAIL 1: BINARIZED D1 SEGMENTS",show_image)
                cv.waitKey(0)


        # if( not pre_bit_pass ):
        #         decode_image = src_atleast_grys
        #         pre_bit_pass = True
        #         enc_bit_encoding = []
        #         id_bit_encoding = []
        #         segment_percentw_vec = []
        #         segment_id_percentw_vec = []
        #         _,decode_image_tzthresh = cv.threshold( 255-decode_image, 255-dcdc.DECODING_GREYSCALE_THRESH, 255, cv.THRESH_TOZERO)
        #         decode_image_tzthresh = 255 - decode_image_tzthresh
        #         decode_blur = cv.medianBlur( decode_image_tzthresh, 3 )
        #         for ii in range(0,dcdc.ENCODING_LENGTH):
        #                 for jj in range(0,dcdc.ENCODING_LENGTH):
        #                         indx = ii*dcdc.ENCODING_LENGTH + jj
        #                         if( indx == cid_indx): 
        #                                 enc_bit_encoding.append(-1) 
        #                                 id_bit_encoding.append(-1) 
        #                                 continue
        #                         segmentSubMatrix = decode_blur[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]/255
        #                         _,sSM_thresh_bin = cv.threshold( segmentSubMatrix.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE )
                                
        #                         if( indx == 0 ):
        #                                 indx0_img = sSM_thresh_bin
        #                         elif( indx == 1 ):
        #                                 indx1_img = sSM_thresh_bin
        #                         elif( indx == 2 ):
        #                                 indx2_img = sSM_thresh_bin
        #                         elif( indx == 3 ):
        #                                 indx3_img = sSM_thresh_bin
        #                         elif( indx == 4 ):
        #                                 indx4_img = sSM_thresh_bin
        #                         elif( indx == 5 ):
        #                                 indx5_img = sSM_thresh_bin
        #                         elif( indx == 6 ):
        #                                 indx6_img = sSM_thresh_bin
        #                         elif( indx == 7 ):
        #                                 indx7_img = sSM_thresh_bin
        #                         elif( indx == 8 ):
        #                                 indx8_img = sSM_thresh_bin

        #                         sSM_thresh_bin = sSM_thresh_bin/255

        #                         segment_percentw = 0.0
        #                         segment_id_percentw = 0.0
        #                         for kk in range(0,2):
        #                                 for ll in range(0,2):
        #                                         segmentSubSubMatrix = sSM_thresh_bin[ kk*row_sub_seg:(kk+1)*row_sub_seg-1, ll*col_sub_seg:(ll+1)*col_sub_seg-1 ]
        #                                         if( kk*2 + ll != cid_corner_indx):
        #                                                 segment_percentw = segment_percentw + cv.sumElems(segmentSubSubMatrix)[0]/( 3*sub_segment_area )
        #                                         else:
        #                                                 segment_id_percentw = segment_id_percentw + cv.sumElems(segmentSubSubMatrix)[0]/( sub_segment_area )
                                
        #                         segment_percentw_vec.append( segment_percentw )
        #                         segment_id_percentw_vec.append( segment_id_percentw )


        #                         if( abs(segment_percentw - 0.5) > dcdc.DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
        #                                 if( (segment_percentw - 0.5) < 0 ):
        #                                         enc_bit_encoding.append(0)
        #                                 else:
        #                                         enc_bit_encoding.append(1)
        #                         else:
        #                                 enc_bit_encoding.append(-2)
        #                                 pre_bit_pass = False


        #                         if( abs(segment_id_percentw - 0.5) > dcdc.DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
        #                                 if( (segment_id_percentw - 0.5) < 0 ):
        #                                         id_bit_encoding.append(0)
        #                                 else:
        #                                         id_bit_encoding.append(1)
        #                         else:
        #                                 id_bit_encoding.append(-2)
        #                                 pre_bit_pass = False


        # if(debug_mode):
        #         print('[DEBUG]: ENCODING INFORMATION FOR 2ND TRIAL:')
        #         print('[DEBUG]: ENCODED SEGMENT: ' + str(enc_bit_encoding) )
        #         print('[DEBUG]: IDENTIFIER SEGMENT: ' + str(id_bit_encoding) )

        # if(debug_mode):
        #         show_image = src_atleast_grys.copy()
        #         for ii in range(0,dcdc.ENCODING_LENGTH):
        #                 for jj in range(0,dcdc.ENCODING_LENGTH):
        #                         indx = ii*dcdc.ENCODING_LENGTH + jj
        #                         if( indx == cid_indx): 
        #                                 continue
        #                         elif( indx == 0 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx0_img
        #                         elif( indx == 1 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx1_img
        #                         elif( indx == 2 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx2_img
        #                         elif( indx == 3 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx3_img
        #                         elif( indx == 4 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx4_img
        #                         elif( indx == 5 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx5_img
        #                         elif( indx == 6 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx6_img
        #                         elif( indx == 7 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx7_img
        #                         elif( indx == 8 ):
        #                                 show_image[ ii*row_seg:(ii+1)*row_seg-1 , jj*col_seg:(jj+1)*col_seg-1 ] = indx8_img

        #         cv.imshow("TRAIL 2: BINARIZED D1 SEGMENTS",show_image)
        #         cv.waitKey(0)


        pre_bit_encoding = []
        for ii in range( 0, len(enc_bit_encoding) ):
                if( (id_bit_encoding[ii] == 0 or id_bit_encoding[ii] == 1) and (enc_bit_encoding[ii] == 0 or enc_bit_encoding[ii] == 1) ):
                        if( enc_bit_encoding[ii] == 1 and id_bit_encoding[ii] == 0 ):
                                pre_bit_encoding.append(1)
                        elif( enc_bit_encoding[ii] == 0 and id_bit_encoding[ii] == 1 ):
                                pre_bit_encoding.append(0)
                        else:
                                pre_bit_encoding.append(-2) 
                                pre_bit_pass = False 
                        
                elif( id_bit_encoding[ii] == -1 and enc_bit_encoding[ii] == -1 ):
                        pre_bit_encoding.append(-1)

                else:
                        pre_bit_encoding.append(-2)
                        pre_bit_pass = False

        if(debug_mode):
                print('\n[DEBUG]: PREBIT ENCODING:' + str(pre_bit_encoding) )
        
        return pre_bit_encoding, pre_bit_pass


def evaluateV1BitEncoding( src_atleast_grys, row_seg, col_seg, segment_area, cid_indx, debug_mode ):
        decode_image = src_atleast_grys
        decode_blur = cv.medianBlur( decode_image, 3 )
        _,decode_thresh_bin = cv.threshold( decode_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE )
        
        pre_bit_pass = True
        pre_bit_encoding = []
        segment_percentw_vec = []
        for ii in range(0,dcdc.ENCODING_LENGTH):
                for jj in range(0,dcdc.ENCODING_LENGTH):
                        if( ii*dcdc.ENCODING_LENGTH + jj == cid_indx): 
                                pre_bit_encoding.append(-1) 
                                continue
                        segmentSubMatrix = decode_thresh_bin[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]/255
                        segment_percentw = cv.sumElems(segmentSubMatrix)[0]/segment_area
                        segment_percentw_vec.append(segment_percentw)

                        if( abs(segment_percentw - 0.5) > dcdc.DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
                                if( (segment_percentw - 0.5) < 0 ):
                                        pre_bit_encoding.append(0)
                                else:
                                        pre_bit_encoding.append(1)
                        else:
                                pre_bit_encoding.append(-2)
                                pre_bit_pass = False


        # if( not pre_bit_pass ):
        #         decode_image = src_atleast_grys
        #         pre_bit_pass = True
        #         pre_bit_encoding = []
        #         segment_percentw_vec = []
        #         _,decode_image_tzthresh = cv.threshold( 255-decode_image, 255-dcdc.DECODING_GREYSCALE_THRESH, 255, cv.THRESH_TOZERO)
        #         decode_image_tzthresh = 255 - decode_image_tzthresh
        #         decode_blur = cv.medianBlur( decode_image_tzthresh, 3 )
        #         _,decode_thresh_bin = cv.threshold( decode_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE )
        #         for ii in range(0,dcdc.ENCODING_LENGTH):
        #                 for jj in range(0,dcdc.ENCODING_LENGTH):
        #                         if( ii*dcdc.ENCODING_LENGTH + jj == cid_indx): 
        #                                 pre_bit_encoding.append(-1) 
        #                                 continue
        #                         segmentSubMatrix = decode_thresh_bin[ ii*row_seg:(ii+1)*row_seg-1, jj*col_seg:(jj+1)*col_seg-1 ]/255
        #                         segment_percentw = cv.sumElems(segmentSubMatrix)[0]/segment_area
        #                         segment_percentw_vec.append(segment_percentw)

        #                         if( abs(segment_percentw - 0.5) > dcdc.DECODING_CONFIDENCE_THRESHOLD - 0.5  ):
        #                                 if( (segment_percentw - 0.5) < 0 ):
        #                                         pre_bit_encoding.append(0)
        #                                 else:
        #                                         pre_bit_encoding.append(1)
        #                         else:
        #                                 pre_bit_encoding.append(-2)
        #                                 pre_bit_pass = False
        
        return pre_bit_encoding, pre_bit_pass


def readMappedEncoding( cid_indx, pre_bit_encoding ):
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
        
        return bit_encoding


def determinePose(transformation_mat, fc_median_distance, circle_indx):
        ## TODO IMPROVE CIRCLE INDEX DETERMINATION

        inv_perspective_matrix = np.linalg.inv(transformation_mat)
        cartesian_points = []
        # ARRANGE POINTS IN ORDER TL TR BR BL
        
        if circle_indx == 0: 
                homogeneous_points = [np.array([0, 0, 1]), \
                              np.array([fc_median_distance,0,1]), \
                              np.array([fc_median_distance,fc_median_distance,1]), \
                              np.array([0,fc_median_distance,1])]
        elif circle_indx == 1:
                homogeneous_points = [np.array([fc_median_distance,0,1]), \
                              np.array([fc_median_distance,fc_median_distance,1]), \
                              np.array([0,fc_median_distance,1]), \
                              np.array([0,0,1])]
        elif circle_indx == 2:
                homogeneous_points = [np.array([0,fc_median_distance,1]), \
                              np.array([0,0,1]), \
                              np.array([fc_median_distance,0,1]), \
                              np.array([fc_median_distance,fc_median_distance,1])]
        elif circle_indx == 3:
                homogeneous_points = [np.array([fc_median_distance,fc_median_distance,1]), \
                              np.array([0,fc_median_distance,1]), \
                              np.array([0,0,1]), \
                              np.array([fc_median_distance,0,1])]

        # TRANSFORMATION 2D WARPED -> 2D UNWARPED  
        for ii in range(0,4):
                # 2D WARPED (HOMOGENEOUS) -> 2D UNWARPED (HOMOGENOUS)
                homogeneous_point = np.dot(inv_perspective_matrix, homogeneous_points[ii])
                # HOMOGENEOUS -> CARTESIAN
                cartesian_point = (homogeneous_point[0] / homogeneous_point[2], homogeneous_point[1]/ homogeneous_point[2])
                cartesian_points.append(cartesian_point)
        
        # SOLVING TVEC AND RVEC
        object_points = np.array(   [   [-fc_median_distance, fc_median_distance, 0], \
                                        [fc_median_distance, fc_median_distance, 0], \
                                        [fc_median_distance, -fc_median_distance, 0], \
                                        [-fc_median_distance, -fc_median_distance, 0]    ],\
                                        dtype = "float32")
        camera_points = np.array([[cartesian_points[0][0], cartesian_points[0][1]],
                          [cartesian_points[1][0], cartesian_points[1][1]],
                          [cartesian_points[2][0], cartesian_points[2][1]],
                          [cartesian_points[3][0], cartesian_points[3][1]]],
                         dtype="float32")
        _, rvec, tvec = cv.solvePnP(object_points, camera_points, dcdc.CAMERA_MATRIX, dcdc.DISTANCE_COEFFICIENTS)

        return rvec, tvec


def findMarkerCentroid(rect_contour_centroids, rect_contour_angles, debug_mode):
        if len(rect_contour_centroids) == 4:
                avg_x, avg_y, total_x, total_y,  = 0,0,0,0
                #FIND CENTROID AVG
                for ii in range(0,len(rect_contour_centroids)):
                        total_x += rect_contour_centroids[ii][0] 
                        total_y += rect_contour_centroids[ii][1] 
                avg_x = total_x / len(rect_contour_centroids)
                avg_y = total_y / len(rect_contour_centroids)
                return (avg_x,avg_y)
        elif len(rect_contour_centroids) == 3 or len(rect_contour_centroids) == 2:
                parallel_flag = False
                angles = []
                #ONLY TWO CONTOURS ARE NEEDED
                parallel_flag = determineIfParallel(rect_contour_angles[0], rect_contour_angles[1])

                if parallel_flag == True: 
                        return  ((rect_contour_centroids[0][0]+rect_contour_centroids[1][0])/2,
                                        (rect_contour_centroids[0][1]+rect_contour_centroids[1][1])/2)
                else:
                        slopes = []
                        y_intercepts = []
                        # y = mx + b
                        for angle in rect_contour_angles:
                                #PERPENDICULAR SLOPE
                                angle += 90
                                slopes.append(math.tan(math.radians(angle)))
                        for ii in range(0,len(slopes)):
                                #  y-mx = b
                                y_intercepts.append(rect_contour_centroids[ii][1] - (slopes[ii] * rect_contour_centroids[ii][0]))
                        # m1x1 + b1 = m2x2 + b2
                        # m1x1 - m2x2 = b2- b1
                        # x1(m1-m2) = b2 - b1
                        # x1 = b2-b1 / m1-m2
                        y_int_difference = y_intercepts[1] - y_intercepts[0]
                        slope_difference = slopes[0] - slopes[1]
                        x_intersection = y_int_difference / slope_difference
                        return (int(x_intersection), int((slopes[0]*x_intersection) + y_intercepts[0]))

def determineD1Corners(image, rect_contours, rect_contour_centroids, rect_angles, debug_mode):
        slopes = []
        contour_outter_vertices = []
        d1_corner_points = [] 
        parallel_flag = False

        # DETERMINE WETHER TO USE PARALLEL OR ORTHOGONAL BAR METHOD
        if len(rect_contours) == 3:
                parallel_flag = False
                # FIND TWO PARALLEL BARS
                for ii in range (0,3):
                        for jj in range(ii+1, 3):
                                if parallel_flag == True:
                                        break
                                if determineIfParallel(rect_angles[ii], rect_angles[jj]) == True:
                                        rect_contours = [rect_contours[ii], rect_contours[jj]]
                                        rect_angles = [rect_angles[ii], rect_angles[jj]]
                                        rect_contour_centroids = [rect_contour_centroids[ii], rect_contour_centroids[jj]]
                                        parallel_flag = True


        elif len(rect_contours) == 2:
                # DETERMINE IF TWO BARS ARE PARALLEL.
                parallel_flag = determineIfParallel(rect_angles[0], rect_angles[1])

        ############################################################# COLLECT RECT CONTOURS OUTTER VERTICES - START
        marker_centroid = findMarkerCentroid(rect_contour_centroids, rect_angles, debug_mode)
        for ii in range (0,2): # FOR EACH RECT CONTOUR
                vertices_pairs = [] 
                rect = cv.minAreaRect(rect_contours[ii])
                rect_vertices = cv.boxPoints(rect).tolist()

                
                closest_point = 1 
                shortest_distance = est.euclideanDistance(rect_vertices[0], rect_vertices[1])
                for jj in range (2,4):
                        distance  = est.euclideanDistance(rect_vertices[0], rect_vertices[jj])
                        if distance < shortest_distance:
                                closest_point = jj
                                shortest_distance = distance
                vertices_pairs.append([rect_vertices[0], rect_vertices[closest_point]])
                rect_vertices.pop(closest_point)
                rect_vertices.pop(0)
                vertices_pairs.append(rect_vertices)
                outter_vertices = []
                for jj in range (0,2):
                        distance1 = est.euclideanDistance(marker_centroid, vertices_pairs[jj][0])
                        distance2 = est.euclideanDistance(marker_centroid, vertices_pairs[jj][1])
                        if distance1 > distance2:
                                outter_vertices.append(vertices_pairs[jj][0])
                        else:
                                outter_vertices.append(vertices_pairs[jj][1])
                contour_outter_vertices.append(outter_vertices)

        # COLLECT RECT CONTOURS OUTTER VERTICES - END 

        if parallel_flag ==  False:
                # TODO MORE TESTING ON POSE ESTIMATION AFTER TRANSPOSE
                rect_contours_y_int = []
                for ii in range (0, 2): # for each contour

                        # GET PERPENDICULAR SLOPE
                        rect_angles[ii] += 90
                        slopes.append(math.tan(math.radians(rect_angles[ii])))
                        # GET Y INTERCEPT
                        y_intercepts = []
                        for jj in range(0,2): 
                                #  y-mx = b
                                y_intercepts.append(contour_outter_vertices[ii][jj][1] - (slopes[ii] * contour_outter_vertices[ii][jj][0]))
                        rect_contours_y_int.append(y_intercepts)

                for ii in range (0,2): # FOR EACH POINT IN CONTOUR1
                        for jj in range (0,2): # FOR EACH POINT IN CONTOUR2
                                # m1x1 + b1 = m2x2 + b2
                                # m1x1 - m2x2 = b2- b1
                                # x1(m1-m2) = b2 - b1
                                # x1 = b2-b1 / m1-m2
                                y_int_difference = rect_contours_y_int[1][jj] - rect_contours_y_int[0][ii]
                                slope_difference = slopes[0] - slopes[1]
                                x_intersection = y_int_difference / slope_difference
                                # y = mx + b
                                d1_corner_points.append( (int(x_intersection), int((slopes[0]*x_intersection) + rect_contours_y_int[0][ii])))
        else:
                
                ################################################################################# PAIR POINTS THAT WILL LIE ON THE SAME LINE - START
                same_slope_points = []
                for ii in range (0,2): 
                        edist1 = est.euclideanDistance(contour_outter_vertices[0][ii], contour_outter_vertices[1][0])
                        edist2 = est.euclideanDistance(contour_outter_vertices[0][ii], contour_outter_vertices[1][1])
                        if edist1 < edist2:
                                same_slope_points.append([contour_outter_vertices[0][ii], contour_outter_vertices[1][0]])
                        else: 
                                same_slope_points.append([contour_outter_vertices[0][ii], contour_outter_vertices[1][1]])

                #  PAIR POINTS THAT WILL LIE ON THE SAME LINE - END

                ################################################################################# FIND CORNER POINTS - START

                for ii in range (0, 2): # FOR EACH POINT PAIR
                        
                        # FIND SLOPE 
                        x_diff = same_slope_points[ii][1][0]-same_slope_points[ii][0][0]
                        y_diff = same_slope_points[ii][1][1]-same_slope_points[ii][0][1]
                        slope = y_diff/x_diff

                        # FIND Y-INTERCEPT
                        y_intercept = 0
                        for jj in range(0,2): # FARTHEST POINT 1 AND 2 AND FIND THE Y INT
                                # y-mx = b
                                # y intercept for each pair
                                y_intercept = same_slope_points[ii][0][1] - (slope * same_slope_points[ii][0][0])

                        # CALCULATE NECESSARY DISTANCES FROM SAME SLOPE POINTS

                        d1 = est.euclideanDistance(same_slope_points[ii][0], same_slope_points[ii][1]) # DIST BETWEEN OUTTER POINTS
                        d2 = d1 * dcdc.OUTTER_POINT_INNER_CORNER_RATIO                                 # DIST BETWEEN ESTIMATED CORNER POINTS
                        d3 = (d1 - d2) / 2                                                             # DIST BETWEEN OUTTER POINT AND CLOSEST INNER POINT

                        angle = math.atan(y_diff/x_diff)
                        deltaX = d3 * math.cos(angle)
                        # EACH POINT PRODUCES TWO POINTS, KEEP THE ONE THE ONE THAT IS CLOSER
                        for jj in range (0,2): # FOR EACH POINT IN THE PAIR
                                y1 = slope * (same_slope_points[ii][jj][0] + deltaX) + y_intercept
                                y2 = slope * (same_slope_points[ii][jj][0] - deltaX) + y_intercept
                                dist1 = est.euclideanDistance(marker_centroid, (same_slope_points[ii][jj][0] + deltaX, y1))
                                dist2 = est.euclideanDistance(marker_centroid, (same_slope_points[ii][jj][0] - deltaX, y2))
                                if dist1 < dist2:
                                        d1_corner_points.append([same_slope_points[ii][jj][0] + deltaX, y1])
                                else:
                                        d1_corner_points.append([same_slope_points[ii][jj][0] - deltaX, y2])

        if debug_mode:
                        dt.showPointsOnImage(d1_corner_points,"CORNER POINTS ESTIMATION BASED ON 2 BARS", image)
                        cv.waitKey(0)
        return d1_corner_points
                        


def determineWarpedImageFrom4Corners(image, rect_corners, debug_mode):
        fc_relative_angles = []
        quadrant0_flag = False
        quadrant3_flag = False
        for ii in range(1,4):
                anglei = est.angleBetweenPoints(rect_corners[0],rect_corners[ii])
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
        
        fc_median_distance =  est.euclideanDistance( rect_corners[0],rect_corners[fc_median_indx])

        ordered_centroids = np.array( [ [ rect_corners[0], rect_corners[fc_ra_min_indx], \
                                          rect_corners[fc_median_indx], rect_corners[fc_ra_max_indx] ] ] , \
                                      dtype = "float32" )
        
        perspective_transformed_centroids = np.array( [ [ ( 0.0, 0), ( fc_median_distance, 0.0), \
                                                          ( fc_median_distance, fc_median_distance), ( 0, fc_median_distance) ] ], \
                                                          dtype = "float32" )
        
        transformationMat = cv.getPerspectiveTransform(ordered_centroids, perspective_transformed_centroids)
        warped_image = cv.warpPerspective(image, transformationMat, (int(fc_median_distance), int(fc_median_distance)))
        
        return warped_image,fc_median_distance, transformationMat

def determineIfParallel(rect_contour_angle1, rect_contour_angle2):
        #CONVERT INTO POSITIVE ANGLE THAT IS LESS THAN 180 DEGREES 
        if rect_contour_angle1 < 0:
                rect_contour_angle1 +=360
        if rect_contour_angle1 >= 180:
                rect_contour_angle1 -=180
        if rect_contour_angle2 < 0:
                rect_contour_angle2 +=360
        if rect_contour_angle2 >= 180:
                rect_contour_angle2 -=180        
       
        difference = abs(rect_contour_angle1-rect_contour_angle2)
        #IF DIFFERENCE IS CLOSE TO 180 OR CLOSE TO 0 THEN PASS
        if (difference <= dcdc.RECT_PARALLEL_ANGLE_THRESH or difference > 180 - dcdc.RECT_PARALLEL_ANGLE_THRESH): 
                return True
        else:
                return False