#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <debugTools.h>
#include <euclideanSpaceTools.h>
using namespace cv;


/// DEBUG PROPERTIES
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DECODER_DEBUG_MODE 0
// 0 - NONE
// 1 - DEBUGGER MODE FOR D1 EXTRACTION
// 2 - DEBUGGER MODE FOR ENCODING DETERMINATION
// 3 - DEBUGGER MODE FOR BOTH MODES (1)(2)
#define DECODER_VERBOSE_MODE false
#define DECODER_SHOWCASE_MODE true
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/// MEASURED PROPERTIES 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // FOR COMPUTER GENERATED IMAGES
// #define ENCODING_LENGTH 3
// #define MEASURED_ASPECT_RATIO 9.615
// #define ENCODING_CROP_RATIO 0.1772151899
// #define SEGMENT_LCIRC_RATIO 0.7853981634
// #define SEGMENT_SCIRC_RATIO 0.0872664626

// // FOR IRL IMAGES
#define ENCODING_LENGTH 3
#define MEASURED_ASPECT_RATIO 6.0
#define ENCODING_CROP_RATIO 0.1666666667
#define SEGMENT_LCIRC_RATIO 0.7853981634
#define SEGMENT_SCIRC_RATIO 0.0872664626
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/// TUNING PARAMETERS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define GREYSCALE_TO_255_THRESHOLD 200

#define RECT_AREA_PERCENT_THRESHOLD 0.15
#define RECT_ASPECT_RATIO_LOWER_PERECNT_ERROR_THRESHOLD 20
#define RECT_ASPECT_RATIO_UPPER_PERECNT_ERROR_THRESHOLD 40
#define RECT_IDENTIFIER_SIDES_LOWER_THRESHOLD 4
#define RECT_IDENTIFIER_SIDES_UPPER_THRESHOLD 6

#define CONTOUR_ED_THRES 20

#define CIRC_AREA_LOWER_PERCENT_THRESHOLD 30
#define CIRC_AREA_UPPER_PERCENT_THRESHOLD 10
#define CIRC_IDENTIFIER_SIDES_THRESHOLD 10

#define DECODING_CONFIDENCE_THRESHOLD 0.65
#define DECODING_GREYSCALE_THRESH 150

// MORPHING PROPERTIES
#define GRADIENT_MORPH_SIZE 1
#define CLOSING_MORPH_SIZE 0
#define MEDIAN_BLUR_SIZE 1
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// FUNCTIONALITY
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace Classification_Filter{

    void determineWarpedImageFrom4IdBars( Mat& image, Mat& warped_image, double& fc_median_distance, std::vector<Point2f>& rect_contour_centroids, bool& DEBUG_MODE)
    {   
        // fc_ -> RELATIVE TO FIRST INDEXED ELEMENT
        std::vector<double> fc_relative_angles = {};
        bool quadrant0_flag = false;
        bool quadrant3_flag = false;
        for( size_t ii = 1; ii < 4; ii++ ){
            double anglei = angleBetweenPoints(rect_contour_centroids[0],rect_contour_centroids[ii]);
            if( determineAngleQuadrant(anglei) == 0 ) quadrant0_flag = true;
            else if( determineAngleQuadrant(anglei) == 3 ) quadrant3_flag = true;
            fc_relative_angles.push_back(anglei);
        }

        if( quadrant0_flag && quadrant3_flag ){
            // IF ELEMENTS EXIST IN BOTH 0-3 QUADRANTS MOVE 
            // THE ANGLES TO THE 1-2 QUADRANTS FOR NEXT STEPS
            for(size_t ii = 0; ii < 3; ii++ ){
                double anglei = fc_relative_angles[ii] + M_PI; 
                if( anglei > 2*M_PI ) anglei = anglei - 2*M_PI;
                fc_relative_angles[ii] = anglei;
            }
        }

        // DETERMINE MIN/MAX RELATIVE ANGLES
        double fc_ra_max = fc_relative_angles[0];
        double fc_ra_min = fc_ra_max;
        size_t fc_ra_max_indx = 1;
        size_t fc_ra_min_indx = 1;
        for( size_t ii = 2; ii < 4; ii++ ){
            double anglei = fc_relative_angles[ii-1];
            if( anglei < fc_ra_min ){
                fc_ra_min = anglei;
                fc_ra_min_indx = ii;
            }
            else if( anglei > fc_ra_max ){ 
                fc_ra_max = anglei;
                fc_ra_max_indx = ii;
            }
        }

        // DETERMINE MEDIAN ELEMENT (ELEMENT THAT SITS BETWEEN THE OTHERS RELATIVE TO fc)
        size_t fc_median_indx = 0;
        for( size_t ii = 2; ii < 4; ii++ ){
            if( ii != fc_ra_min_indx && ii != fc_ra_max_indx ){
                fc_median_indx = ii; // INDEX IN RELATIVE SET
            }
        }

        // DETERMINE MEDIAN DISTANCE FOR CROPPING
        fc_median_distance =  euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[fc_median_indx]);
        

        // DETERMINE WHERE CENTROIDS SHOULD BE AFTER PERSPECTIVE TRANSFORM
        std::vector<Point2f> ordered_centroids = { rect_contour_centroids[0], rect_contour_centroids[fc_ra_min_indx], 
                    rect_contour_centroids[fc_median_indx], rect_contour_centroids[fc_ra_max_indx] };

        std::vector<Point2f> perspective_transformed_centroids = { 
                    Point2f( 0, 0.5*fc_median_distance), 
                    Point2f( 0, 0.5*fc_median_distance) + Point2f( 0.5*fc_median_distance, -0.5*fc_median_distance),
                    Point2f( 0, 0.5*fc_median_distance) + Point2f(fc_median_distance,0) ,
                    Point2f( 0, 0.5*fc_median_distance) + Point2f( 0.5*fc_median_distance, 0.5*fc_median_distance)
                };
        
        Size warped_image_size = Size(cvRound(fc_median_distance), cvRound(fc_median_distance));
        Mat transformationMat = getPerspectiveTransform(ordered_centroids, perspective_transformed_centroids);
        // WARP THE SOURCE AS TO AVOID ANY CHANGES BEING NECCESSARY FOR SECONDARY MODULE
        warpPerspective(image, warped_image, transformationMat, warped_image_size);
        
        // MAKE-SHIFT DEBUG SECTION    
        if(DEBUG_MODE){
            if(DECODER_VERBOSE_MODE){
                std::cout << "[DEBUG] RECT CENTROID 1: " << rect_contour_centroids[0] << std::endl;
                std::cout << "[DEBUG] RECT CENTROID 2: " << rect_contour_centroids[fc_ra_max_indx] << std::endl;
                std::cout << "[DEBUG] RECT CENTROID 3: " << rect_contour_centroids[fc_median_indx] << std::endl;
                std::cout << "[DEBUG] RECT CENTROID 4: " << rect_contour_centroids[fc_ra_min_indx] << std::endl;
            }
            imshow( "D2 CROPPED IMAGE", warped_image); 
            waitKey();
        }

        return;
    }


    void determineWarpedImageFrom2or3IdBars( Mat& image, Mat& warped_image, double& fc_median_distance, std::vector<Point2f>& rect_contour_centroids, 
         std::vector<double>& rect_contour_angles, bool& DEBUG_MODE)
    {   
        // DETERMINE REFERENCE ANGLE FROM FIRST INDEX
        double ref_angle = rect_contour_angles[0];

        // FIT A LINE THROUGH REFERENCE RECT (WHICH IS PARALELL TO THE LONGEST SIDE)
        double slope = std::tan ( ref_angle * M_PI / 180.0 );
        double yinter = rect_contour_centroids[0].y - slope*rect_contour_centroids[0].x;
        
        /// \TODO: CHECK ALL ANGLES AND PERFORM THE MEAN CALCULATED ROTATION 
        /// BY GEOMETRIC RESTRICTIONS, IF ONE ELEMENT IS ABOVE THE LINE DESCRIBED ABOVE, ALL OTHERS ARE
        bool above_reference_line = rect_contour_centroids[1].y - (slope*rect_contour_centroids[1].x + yinter) > 0;

        // DETERMINE MEDIAN DISTANCE BASED ON A PREDICTION OF WHETHER THE OTHER BAR IS (RELATIVELY) THE PARALELL OR NORMAL BAR 
        if( fabs(ref_angle - rect_contour_angles[0]) > 45.0 ){
            fc_median_distance = 2 * euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1]) / std::sqrt(2);
        }
        else fc_median_distance = euclideanDistance( rect_contour_centroids[0],rect_contour_centroids[1]);

        // DETERMINE GEOMETRIC TRANSFORMATION 
        double delta_angle = 0;
        if(above_reference_line) delta_angle = -90 - ref_angle;
        else delta_angle = 90 - ref_angle;

        Mat rotatedImage;
        Size rotatedimageSize;
        rotatedimageSize.height = 2 * image.size().height;
        rotatedimageSize.width = 2 * image.size().width;

        Mat transformationMat = getRotationMatrix2D( rect_contour_centroids[0], -delta_angle, 1 );
        warpAffine(image, rotatedImage, transformationMat, rotatedimageSize );
        warped_image = rotatedImage(
            Range( cvRound( rect_contour_centroids[0].y - 0.5*fc_median_distance  ), cvRound( rect_contour_centroids[0].y + 0.5*fc_median_distance  )),
            Range( cvRound( rect_contour_centroids[0].x ), cvRound( rect_contour_centroids[0].x + fc_median_distance  ))
        );

        // MAKE-SHIFT DEBUG SECTION          
        if(DEBUG_MODE){
            if(DECODER_VERBOSE_MODE){
                std::cout << "[DEBUG] WARPED IMAGE SIZE: "<<  warped_image.size << std::endl;
                std::cout << "[DEBUG] MERIDIAN LENGTH: "<<  fc_median_distance << std::endl;
                std::cout << "[DEBUG] REFERENCE CENTROID: "<<  rect_contour_centroids[0] << std::endl;
                std::cout << "[DEBUG] REFERENCE ANGLE: "<<  ref_angle << std::endl;
                if( above_reference_line ) std::cout << "[DEBUG] ELEMENTS ABOVE REF LINE " << std::endl;
                else std::cout << "[DEBUG] ELEMENTS BELOW REF LINE " << std::endl;
                std::cout << "[DEBUG] DELTA ANGLE: "<<  delta_angle << std::endl;
            }
            imshow( "D3 ROTATED IMAGE (WARPED IMAGE)", rotatedImage); 
            imshow( "D2 CROPPED IMAGE", warped_image); 
            waitKey();
        }

        return;
    }


    Mat extractD1Domain( Mat& image, bool& DEBUG_MODE ){
        
        // PBR VARIABLES - START
        std::vector<std::vector<Point> > contours;
        std::vector<Vec4i> hierarchy;
        std::vector<std::vector<Point> > rect_contours,rect_contours_corr,rect_contours_pass1;
        std::vector<Point2f> rect_contour_centroids,rect_contour_centroids_corr;
        std::vector<double> rect_contour_angles,rect_contour_angles_corr;
        Mat src_gray, src_thresh_prelim, src_blur, src_eval_contours, canny_output;
        // PBR VARIABLES - END

        // SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - START
        cvtColor( image, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
        threshold(255-src_gray, src_thresh_prelim, 255-GREYSCALE_TO_255_THRESHOLD, 255, THRESH_TOZERO); 
        medianBlur( 255-src_thresh_prelim, src_blur, 2*int( MEDIAN_BLUR_SIZE )+1);

        Mat element1 = getStructuringElement( 
            2, Size( 2*int( CLOSING_MORPH_SIZE ) + 1, 2*int( CLOSING_MORPH_SIZE )+1 ), Point( int( CLOSING_MORPH_SIZE ), int( CLOSING_MORPH_SIZE ) ) 
        );
        morphologyEx( src_blur, src_eval_contours, 3, element1 );  
        imshow("checking",src_blur); waitKey();
        // SETTING UP APPROPRIATE SETTING FOR CONTOUR DETERMINATION - END


        // FIND LOCATING BARS - START
        Canny( src_eval_contours, canny_output, 10, 200 );
        findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_KCOS ); 
        if(DEBUG_MODE) std::cout << "\n" << std::endl;
        for( size_t ii = 0; ii < contours.size(); ii++ )
        {
            std::vector<Point> approx_cp;
            approxPolyDP( contours[ii], approx_cp, 0.025*arcLength(contours[ii],true), true );
            double approx_cp_area = fabs( contourArea(approx_cp) );
            
            if(DEBUG_MODE && DECODER_VERBOSE_MODE){
                std::cout << "[DEBUG] Analyzing RECT Contour: " << ii << std::endl;
                std::cout << "[DEBUG] Contour Area: " << approx_cp_area << std::endl;
                std::cout << "[DEBUG] Contour Area Threshold: " << image.rows * image.cols * float(RECT_AREA_PERCENT_THRESHOLD)/100  << std::endl;
                std::cout << "[DEBUG] Contour Sides: " << approx_cp.size() << std::endl;
            } 

            if( (    approx_cp_area > image.rows * image.cols * float(RECT_AREA_PERCENT_THRESHOLD)/100 )                    // 1. AREA PROTECTION
                && approx_cp.size() >= RECT_IDENTIFIER_SIDES_LOWER_THRESHOLD 
                && approx_cp.size() <= RECT_IDENTIFIER_SIDES_UPPER_THRESHOLD  )    // EXPECTING EXACT RECTANGULAR APPROXIMATION IS ASKING TOO MUCH                                                                         // 2. POLY-SIDE PROTECTION
            {
                
                // I'D PREFER NOT TO DO THE FOLLOWING UNLESS (1)(2) PROTECTIONS PASS
                rect_contours_pass1.push_back( contours[ii]  );
                RotatedRect approx_rect = minAreaRect(approx_cp);
                double approx_aspectr = std::max( approx_rect.size.height/approx_rect.size.width, approx_rect.size.width/approx_rect.size.height );
                //if(DEBUG_MODE) std::cout << "[DEBUG] Contour ASPECT RATIO: " << approx_cp.size() << std::endl;
                if(    approx_aspectr > (1 - float(RECT_ASPECT_RATIO_LOWER_PERECNT_ERROR_THRESHOLD)/100) * MEASURED_ASPECT_RATIO    // 3. ASPECT RATIO PROTECTION
                    && approx_aspectr < (1 + float(RECT_ASPECT_RATIO_UPPER_PERECNT_ERROR_THRESHOLD)/100) * MEASURED_ASPECT_RATIO )
                {
                    rect_contours.push_back( contours[ii] );
                    Moments contour_moments = moments( contours[ii], false );
                    Point2f contour_centroid = Point2f( (contour_moments.m10/contour_moments.m00) , (contour_moments.m01/contour_moments.m00) );
                    rect_contour_centroids.push_back( contour_centroid );
                    if( approx_rect.size.height/approx_rect.size.width < 1 ) rect_contour_angles.push_back( approx_rect.angle );
                    else{ 
                        int anglesign = int( approx_rect.angle > 1 ) - int( approx_rect.angle < 1);
                        rect_contour_angles.push_back( - anglesign * ( 90 - fabs(approx_rect.angle) ) );;
                    }
                    rect_contour_angles.push_back(approx_rect.angle);
                    if(DEBUG_MODE && DECODER_VERBOSE_MODE){
                        std::cout << "[DEBUG] Contour RECT angle: " <<   approx_rect.angle << std::endl;
                        std::cout << "[DEBUG] Contour RECT Centroid: " <<   contour_centroid << std::endl;
                        std::cout << "[DEBUG] Contour RECT size: " <<   approx_rect.size << "\n"<< std::endl;
                    }
                }
            }
            if(DEBUG_MODE && DECODER_VERBOSE_MODE) std::cout << "[DEBUG] Finished RECT Contour: " << ii << std::endl;

        }
        if(DEBUG_MODE && DECODER_VERBOSE_MODE) std::cout << "\n" << std::endl;
        // FIND LOCATING BARS - END

        if(DEBUG_MODE){
            imshow( "D3 IMAGE USED TO CALCULATE CONTOURS", src_eval_contours); 
            showContours(contours,hierarchy,"CONTOURS IN D3",image);
            showContours(rect_contours_pass1,hierarchy,"CONTOURS THAT PASS EVERYTHING BUT ASPECT-RATIO",image);
            waitKey();
        }

        if(DECODER_SHOWCASE_MODE){
            showContoursAndCentersOnImage(rect_contours,hierarchy,rect_contour_centroids,"PASSING CONTOURS ON IMAGE",image); waitKey();
        }
        
        
        // PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - START    
        if( rect_contour_centroids.size()>4 ){
            if(DEBUG_MODE){
                std::cout << "[WARNING] FOUND: "<< rect_contour_centroids.size() << " IDENTIFING BARS"<< std::endl;
                if(rect_contour_centroids.size()>0 ){ showContours(rect_contours,hierarchy,"TOO MANY IDENTIFYING BARS",image); waitKey();}; 
            }
            
            if( rect_contour_centroids.size() > 4 ){

                rect_contour_centroids_corr.push_back( rect_contour_centroids[0] );
                for( int ii = 1; ii < rect_contour_centroids.size(); ii++ ){
                    
                    bool pass_flag = true;
                    for( int jj = 0; jj < rect_contour_centroids_corr.size(); jj++ ){
                        if( euclideanDistance( rect_contour_centroids[ii] , rect_contour_centroids_corr[jj] ) <= CONTOUR_ED_THRES ){ 
                            pass_flag = false; 
                            break; 
                        } 
                    }
                    
                    if( pass_flag ){
                        rect_contour_centroids_corr.push_back( rect_contour_centroids[ii] );
                        rect_contour_angles_corr.push_back( rect_contour_angles[ii] );
                    }

                }
                rect_contour_centroids = rect_contour_centroids_corr;
                rect_contour_angles = rect_contour_angles_corr;
                if(DEBUG_MODE) std::cout << "[WARNING] ID BAR CORRECTION ATTEMPTED: "<< rect_contour_centroids.size() << std::endl;
            }

            if( rect_contour_centroids.size() > 4 ){
                if(DEBUG_MODE){
                    std::cout << "[ERROR] ID BAR CORRECTION FAILED AS THERE ARE >4 ID-BARS: "<< rect_contour_centroids.size() << std::endl << " NOW EXITING \n";
                    if(rect_contour_centroids.size()>0 ){ showContours(rect_contours,hierarchy,"CORRECTED RECTANGULAR CONTOURS IN D3 AND THEIR CENTROIDS",image); waitKey();}; 
                }
                return Mat(); // EMPTY IMAGE
            }
        }
        // PROTECTION AGAINST MORE/LESS THAN FOR 4 BARS - END
        
        Mat warped_image;
        double fc_median_distance = 0.;
        if( rect_contour_centroids.size()==4 ){
            determineWarpedImageFrom4IdBars( image, warped_image, fc_median_distance, rect_contour_centroids, DEBUG_MODE);
        }
        else if( rect_contour_centroids.size() < 4 && rect_contour_centroids.size() > 1){
            determineWarpedImageFrom2or3IdBars( image, warped_image, fc_median_distance, rect_contour_centroids,rect_contour_angles, DEBUG_MODE);
        }else return Mat();


        Range encoding_cropping_range = Range( cvRound(float(ENCODING_CROP_RATIO)*fc_median_distance) ,
            cvRound( (1-float(ENCODING_CROP_RATIO))*fc_median_distance) );
        Mat encodedImage = warped_image( encoding_cropping_range, encoding_cropping_range ); 

        return encodedImage;
    }

    std::vector<bool> determineEncodingFromD1Image( Mat& image, bool& DEBUG_MODE ){
        // BLACK SEGMENT - 0
        // WHITE SEGMENT - 1

        // PBR VARIABLES - START
        Mat src_gray, src_thresh_prelim, src_eval_contours,canny_output, decode_blur, decode_thresh_bin;
        std::vector<std::vector<Point> > contours;
        std::vector<Vec4i> hierarchy;
        std::vector<std::vector<Point> > circ_contours;
        std::vector<Point2f> circ_contour_centroids;
        std::vector<double> segment_percentw_vec = {};
        std::vector<bool> bit_encoding = {};
        // PBR VARIABLES - END

        cvtColor( image, src_gray, COLOR_BGR2GRAY );
        threshold(255-src_gray, src_thresh_prelim, 255-GREYSCALE_TO_255_THRESHOLD, 255, THRESH_TOZERO); 
        src_eval_contours = 255 - src_thresh_prelim;
    
        // SEGMENTATION - START
        int row_seg = cvRound(src_eval_contours.rows/int(ENCODING_LENGTH)); 
        int col_seg = cvRound(src_eval_contours.cols/int(ENCODING_LENGTH));
        float segment_area = src_eval_contours.rows * src_eval_contours.cols/( float(ENCODING_LENGTH) * float(ENCODING_LENGTH) );
        if(DEBUG_MODE && DECODER_VERBOSE_MODE) std::cout << "[DEBUG] ENCODING SEGMENT AREA: " << segment_area << std::endl;
        // SEGMENTATION - END


        // LOCATING CIRCULAR IDENTIFIER - START
        Canny( src_eval_contours, canny_output, 10, 200 );      
        findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_L1 ); // CHECK CONTOUR RETRIEVAL MODES
        Point2f weighted_sum_circ_centroid = Point2f(0.0,0.0);
        double total_weighted_area = 0.0;
        
        for( size_t ii = 0; ii < contours.size(); ii++ )
        {
            if(DEBUG_MODE && DECODER_VERBOSE_MODE) std::cout << "\n[DEBUG] Analyzing for Circ Contour: " << ii << std::endl;
            std::vector<Point> approx_cp;
            approxPolyDP( contours[ii], approx_cp, 0.01*arcLength(contours[ii],true), true );
            if( approx_cp.size() > CIRC_IDENTIFIER_SIDES_THRESHOLD  ){
                Point2f approx_min_circ_center;
                float approx_min_circ_radius;
                minEnclosingCircle( contours[ii], approx_min_circ_center, approx_min_circ_radius );
                float approx_cp_area = M_PI*std::pow(approx_min_circ_radius,2);
                if(    approx_cp_area < ( segment_area ) * (SEGMENT_LCIRC_RATIO) * (1 + float(CIRC_AREA_UPPER_PERCENT_THRESHOLD)/100)                        // 1. AREA PROTECTION
                    && approx_cp_area > ( segment_area ) * (SEGMENT_SCIRC_RATIO) * (1 - float(CIRC_AREA_LOWER_PERCENT_THRESHOLD)/100)  )                                                                               // 2. POLY-SIDE PROTECTION
                {

                    if(DEBUG_MODE && DECODER_VERBOSE_MODE){
                        std::cout << "[DEBUG] Circ Sides: " << approx_cp.size() << std::endl;
                        std::cout << "[DEBUG] Circ Area: " << approx_cp_area << std::endl;
                        std::cout << "[DEBUG] Circ Area Upper Thresh: " << ( segment_area ) * (SEGMENT_LCIRC_RATIO) * (1 + float(CIRC_AREA_UPPER_PERCENT_THRESHOLD)/100)  << std::endl;
                        std::cout << "[DEBUG] Circ Area Lower Thresh: " << ( segment_area ) * (SEGMENT_SCIRC_RATIO) * (1 - float(CIRC_AREA_LOWER_PERCENT_THRESHOLD)/100)  << "\n" <<std::endl;
                    }

                    circ_contours.push_back( contours[ii] );
                    circ_contour_centroids.push_back( approx_min_circ_center );
                    weighted_sum_circ_centroid = weighted_sum_circ_centroid + approx_cp_area*approx_min_circ_center; 
                    total_weighted_area = total_weighted_area + approx_cp_area;
                }
            }
            if(DEBUG_MODE && DECODER_VERBOSE_MODE ) std::cout << "[DEBUG] FINISHED Analyzing for Circ Contour: " << ii << std::endl;
        }

        Point2f avg_circ_centroid = Point2f( 
            weighted_sum_circ_centroid.x/total_weighted_area, weighted_sum_circ_centroid.y/total_weighted_area);

        // DETERMINE BIT ENCODING - START
        std::vector<int> pre_bit_encoding = {}; // CHANGE TO BIT SET

        bool pre_bit_pass = true;
        bool cid_found = false;
        int cid_indx = -1; 
        Mat encoded_src_threshed = src_eval_contours;
        Mat cidSegementSubmatrix;

        for( int ii = 0; ii < ENCODING_LENGTH && !cid_found; ii++){
            for( int jj = 0; jj < ENCODING_LENGTH && !cid_found; jj++){
                if( cvRound(avg_circ_centroid.x) >= jj*col_seg && cvRound(avg_circ_centroid.x) <= (jj+1)*col_seg - 1
                 && cvRound(avg_circ_centroid.y) >= ii*row_seg && cvRound(avg_circ_centroid.y) <= (ii+1)*row_seg - 1 ){
                    cid_indx = ii*ENCODING_LENGTH + jj;
                    cid_found = true;
                    cidSegementSubmatrix = encoded_src_threshed( 
                        Range( ii*row_seg , (ii+1)*row_seg - 1 ) , 
                        Range( jj*col_seg , (jj+1)*col_seg - 1 ) 
                    );
                }
            }
        }


        if( DEBUG_MODE ){
            showContours(contours,hierarchy,"CONTOURS IN D1",image);
            showContoursAndCenters(circ_contours,hierarchy,circ_contour_centroids,"PASSING CIRCULAR CONTOURS IN D1 AND THEIR CENTROIDS",image); 
            imshow("THRESHED IMAGE OF ENCODING", encoded_src_threshed);
            waitKey();
        }

        /// \TODO: ERROR CODE THIS
        if( !cid_found ) return bit_encoding;

        if(DECODER_SHOWCASE_MODE){
            Mat show_image = image;
            showContoursAndCentersOnImage(circ_contours,hierarchy,circ_contour_centroids,"PASSING CIRCULAR CONTOURS IN D1 AND THEIR CENTROIDS",show_image); 
        }

        Mat decode_image = encoded_src_threshed;
        medianBlur( decode_image, decode_blur, 3 );
        threshold( decode_blur, decode_thresh_bin, 0, 255, THRESH_BINARY | THRESH_TRIANGLE );
        if(DEBUG_MODE){ imshow("NO SHADOW REMOVAL BINARY ENCODED IMAGE", decode_thresh_bin ); waitKey();}
        for( int ii = 0; ii < ENCODING_LENGTH; ii++){
            for( int jj = 0; jj < ENCODING_LENGTH; jj++){
                if( ii*ENCODING_LENGTH + jj == cid_indx){ pre_bit_encoding.push_back(-1); continue; }

                Mat segmentSubMatrix = decode_thresh_bin( 
                    Range( ii*row_seg , (ii+1)*row_seg - 1 ) , 
                    Range( jj*col_seg , (jj+1)*col_seg - 1 ) 
                )/255;

                double segment_percentw = sum(segmentSubMatrix)[0]/segment_area;
                segment_percentw_vec.push_back( segment_percentw );
                
                
                if( abs(segment_percentw - 0.5) > DECODING_CONFIDENCE_THRESHOLD - 0.5  ){
                    if( (segment_percentw - 0.5) < 0 ) pre_bit_encoding.push_back(0);
                    else pre_bit_encoding.push_back(1);
                }else{
                    pre_bit_encoding.push_back(-2);
                    pre_bit_pass = false;
                }   
            }
        }

        
        if(!pre_bit_pass){ //TRY THIS IF YOU FAIL
            Mat decode_image_tzthresh;
            decode_image = encoded_src_threshed;
            pre_bit_encoding = {};
            segment_percentw_vec = {};
            pre_bit_pass = true;
            threshold(255-decode_image, decode_image_tzthresh, 255-DECODING_GREYSCALE_THRESH, 255, THRESH_TOZERO); 
            decode_image_tzthresh = 255 - decode_image_tzthresh;
            medianBlur( decode_image_tzthresh, decode_blur, 3 );
            threshold( decode_blur, decode_thresh_bin, 0, 255, THRESH_BINARY | THRESH_TRIANGLE );
            if(DEBUG_MODE){ imshow("ATTEMPTED SHADOW REMOVAL BINARY ENCODED IMAGE", decode_thresh_bin ); waitKey();}
            for( int ii = 0; ii < ENCODING_LENGTH; ii++){
                for( int jj = 0; jj < ENCODING_LENGTH; jj++){
                    if( ii*ENCODING_LENGTH + jj == cid_indx){ pre_bit_encoding.push_back(-1); continue; }

                    Mat segmentSubMatrix = decode_thresh_bin( 
                        Range( ii*row_seg , (ii+1)*row_seg - 1 ) , 
                        Range( jj*col_seg , (jj+1)*col_seg - 1 ) 
                    )/255;

                    double segment_percentw = sum(segmentSubMatrix)[0]/segment_area;
                    segment_percentw_vec.push_back( segment_percentw );
                    
                    
                    if( abs(segment_percentw - 0.5) > DECODING_CONFIDENCE_THRESHOLD - 0.5  ){
                        if( (segment_percentw - 0.5) < 0 ) pre_bit_encoding.push_back(0);
                        else pre_bit_encoding.push_back(1);
                    }else{
                        pre_bit_encoding.push_back(-2);
                        pre_bit_pass = false;
                    }   
                }
            }
        }

        // LOCATING CIRCULAR IDENTIFIER - END
        // DETERMINE BIT ENCODING - CONTINUING

        /// \TODO: GENERALIZE FOR DIFFERENT ENCODING SCHEMES 
        if( pre_bit_pass && cid_found){

            switch (cid_indx)
            {
                case 0:
                    bit_encoding = {
                        bool(pre_bit_encoding[1]),bool(pre_bit_encoding[2]),bool(pre_bit_encoding[3]),bool(pre_bit_encoding[4]),
                        bool(pre_bit_encoding[5]),bool(pre_bit_encoding[6]),bool(pre_bit_encoding[7]),bool(pre_bit_encoding[8])
                    };
                    break;
                
                case 2:
                    bit_encoding = {
                        bool(pre_bit_encoding[5]),bool(pre_bit_encoding[8]),bool(pre_bit_encoding[1]),bool(pre_bit_encoding[4]),
                        bool(pre_bit_encoding[7]),bool(pre_bit_encoding[0]),bool(pre_bit_encoding[3]),bool(pre_bit_encoding[6])
                    };
                    break;

                case 6:
                    bit_encoding = {
                        bool(pre_bit_encoding[3]),bool(pre_bit_encoding[0]),bool(pre_bit_encoding[7]),bool(pre_bit_encoding[4]),
                        bool(pre_bit_encoding[1]),bool(pre_bit_encoding[8]),bool(pre_bit_encoding[5]),bool(pre_bit_encoding[2])
                    };
                    break;

                case 8:
                    bit_encoding = {
                        bool(pre_bit_encoding[7]),bool(pre_bit_encoding[6]),bool(pre_bit_encoding[5]),bool(pre_bit_encoding[4]),
                        bool(pre_bit_encoding[3]),bool(pre_bit_encoding[2]),bool(pre_bit_encoding[1]),bool(pre_bit_encoding[0])
                    };
                    break;
                
                default:
                    // IDENTIFIER NOT IN APPROPRIATE LOCATION
                    break;
            }

        }
        //DETERMINE BIT ENCODING - END

        if( DEBUG_MODE && DECODER_VERBOSE_MODE){
            std::cout << "[DEBUG] SEGMENT LENGTH: " << row_seg << std::endl;
            std::cout << "[DEBUG] AVERAGE CIRCULAR CENTROID: " << avg_circ_centroid << std::endl;
            std::cout << "[DEBUG] CID INDEX: " << cid_indx << std::endl;
            std::cout << "[DEBUG] PREBIT ENCODING: "; print_vec(pre_bit_encoding);
            std::cout << "[DEBUG] SEGMENT CONFIDENCES: "; print_vec(segment_percentw_vec);
        }

        if(DECODER_SHOWCASE_MODE){
            Mat show_image = image;
            showEncodingInformation( pre_bit_encoding, row_seg, ENCODING_LENGTH, "DECODED MESSAGE",  show_image ); waitKey();
        }
        
        return bit_encoding;
    }


    std::vector<bool> decodeImage( Mat& image, const int DEBUG_MODE ){
        // DEBUG MODE: 0 - NO DEBUG
        //             1 - DEBUG D1 EXTRACTION
        //             2 - DEBUG ENCODING READ
        //             3 - DEBUG BOTH
        bool DEBUG_D1E_FLAG = false;
        bool DEBUG_DENC_FLAG = false;

        switch (DEBUG_MODE)
        {
            case 1:
                DEBUG_D1E_FLAG = true;
                DEBUG_DENC_FLAG = false ;
                break;
            
            case 2:
                DEBUG_D1E_FLAG = false;
                DEBUG_DENC_FLAG = true;
                break;

            case 3:
                DEBUG_D1E_FLAG = true;
                DEBUG_DENC_FLAG = true;
                break;
            
            default:
                break;
        }
        
        if( DEBUG_D1E_FLAG || DEBUG_DENC_FLAG ) imshow( "ORIGINAL IMAGE", image );
        if( DEBUG_D1E_FLAG || DEBUG_DENC_FLAG ) std::cout << "[DEBUG] STARTING D1 EXTRACTION" << std::endl << std::flush;
        Mat encodedImage = extractD1Domain(image, DEBUG_D1E_FLAG);
        if( DEBUG_D1E_FLAG || DEBUG_DENC_FLAG ) std::cout << "[DEBUG] D1 EXTRACTION COMPLETE" << std::endl << std::flush;
        if( encodedImage.empty() ) return {};
        else return determineEncodingFromD1Image( encodedImage, DEBUG_DENC_FLAG );
    }

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/// IN-TERMINAL DEPLOYMENT
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv ){

    String imageName("errorerror.jpg"); // by default
    int debugmode = DECODER_DEBUG_MODE;
    if (argc > 1)
    {
        imageName = argv[1];
    }
    if( debugmode > 0  ) std::cout << "[DEBUG] ATTEMPTING IMAGE READ ... " << std::endl;
    Mat src = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Load an image
    if (src.empty())
    {
        std::cout << "[ERROR] CANNOT READ IMAGE: " << imageName << std::endl;
        return -1;
    }else if(debugmode>0) std::cout << "[DEBUG] IMAGE READ: " << imageName << std::endl;

    std::vector<bool> ph = Classification_Filter::decodeImage( src, int(DECODER_DEBUG_MODE) );
    std::cout << "\n \nENCODED MESSGAE: ";
    print_vec(ph);
    std::cout << "\n \n: ";
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////