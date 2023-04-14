import cv2 as cv
import math
import numpy as np
import metrics2D as est
import decodeConst as dcdc
import drawingTools as dt


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
        above_reference_line = ( rect_contour_centroids[1][1] - (slope*rect_contour_centroids[1][0] + yinter) ) > 0

        fc_median_distance = 0.
        if( abs(ref_angle - rect_contour_angles[0]) > 45.0 ):
                fc_median_distance = 2 * est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1]) / math.sqrt(2)
        else:
                fc_median_distance = est.euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1])
        
        delta_angle = 0

        # TODO: TEST THIS DELTA ANGLE DETERMINATION
        if(above_reference_line):
                delta_angle = 90 - ref_angle
        else:
                delta_angle = -90 - ref_angle

        img_height, img_width,_ = image.shape
        rotatedimageSize = ( int(2*img_height) , int(2*img_width)  )

        transformationMat = cv.getRotationMatrix2D( rect_contour_centroids[0], delta_angle, 1 )
        rotatedImage = cv.warpAffine(image, transformationMat, rotatedimageSize )
        warped_image = rotatedImage[ int(rect_contour_centroids[0][1]-0.5*fc_median_distance):int( rect_contour_centroids[0][1]+0.5*fc_median_distance ) , \
                                     int(rect_contour_centroids[0][0]):int(rect_contour_centroids[0][0]+fc_median_distance ) ]
        
        return warped_image, fc_median_distance



def attemptIdBarCorrections( rect_contours , rect_contour_centroids, rect_contour_angles , rect_contour_areas , poly_contour_areas, debug_mode, image_shape ):
        # 1ST PROTECTION FOR MORE THAN 4 BARS (CENTROID PROXIMITY) -START
        rect_contours_corr = []
        rect_contour_centroids_corr = []
        rect_contour_angles_corr = []
        rect_contour_areas_corr = []
        poly_contour_areas_corr = []
        rect_contours_corr.append( rect_contours[0] )
        rect_contour_centroids_corr.append( rect_contour_centroids[0] )
        rect_contour_angles_corr.append( rect_contour_angles[0] )
        rect_contour_areas_corr.append( rect_contour_areas[0] )
        poly_contour_areas_corr.append( poly_contour_areas[0] )

        for ii in range( 1, len(rect_contour_centroids) ):
                pass_flag = True
                
                for jj in range( 0, len(rect_contour_centroids_corr) ):
                        if( est.euclideanDistance( rect_contour_centroids[ii], rect_contour_centroids_corr[jj] ) <= float(dcdc.CONTOUR_ED_THRES) ):
                                # DO A SIZE CHECK TO CHOOSE WHICH TO REPLACE 
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
        if(debug_mode):
                        dt.showContours(rect_contours,"CONTOURS AFTER FIRST PROTECTION",image_shape)
                        cv.waitKey(0)
        # 1ST PROTECTION FOR MORE THAN 4 BARS (CENTROID PROXIMITY) - END


        # 2ND PROTECTION FOR MORE THAN 4 BARS ( RECT APPROX STD METHOD ) - START
        if( len(rect_contour_areas) > 4 and len(rect_contour_centroids) > 1 ):

                # CUTOFF IF TOO MANY 
                if( len(rect_contour_areas) > dcdc.RECT_CUTOFF_SIZE ):
                        sorted_indx = np.argsort(  np.array( rect_contour_areas ) )
                        # TRIM THE NUMBER OF CONTOURS BASED ON AREA
                        rect_contours = [ rect_contours[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        rect_contour_centroids = [ rect_contour_centroids[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        rect_contour_angles = [ rect_contour_angles[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        rect_contour_areas = [ rect_contour_areas[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]
                        poly_contour_areas = [ poly_contour_areas[sorted_indx[ii]] for ii in range(0,dcdc.RECT_CUTOFF_SIZE) ]

                size_before = len(rect_contour_areas)
                while( len(rect_contour_areas) > 4 ):
                        rect_contours_corr = []
                        rect_contour_centroids_corr = []
                        rect_contour_angles_corr = []
                        rect_contour_areas_corr = []
                        poly_contour_areas_corr = []
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

                if(debug_mode):
                        dt.showContours(rect_contours,"CONTOURS AFTER SECOND PROTECTION",image_shape)
                        cv.waitKey(0)
        # 2ND PROTECTION FOR MORE THAN 4 BARS ( RECT APPROX STD METHOD ) - END


        # # 3RD PROTECTION FOR MORE THAN 4 BARS ( CONTOUR TRUE STD METHOD ) - START
        # if( len(rect_contour_areas) > 4 and len(rect_contour_centroids) > 1 ):
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