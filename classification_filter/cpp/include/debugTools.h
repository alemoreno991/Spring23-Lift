#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

using namespace cv;
RNG rng(12345);

void print_vec( std::vector <bool> a) {
   std::cout << '[';
   for(int i=0; i < a.size(); i++) std::cout << a.at(i) << ' ' << std::flush;
   std::cout << ']' <<std::endl;
}
void print_vec( std::vector <int> a) {
   std::cout << '[';
   for(int i=0; i < a.size(); i++) std::cout << a.at(i) << ' ' << std::flush;
   std::cout << ']' <<std::endl;
}
void print_vec( std::vector <double> a) {
   std::cout << '[';
   for(int i=0; i < a.size(); i++) std::cout << a.at(i) << ' ' << std::flush;
   std::cout << ']' <<std::endl;
}

void showContours( std::vector<std::vector<Point> > contoursi, std::vector<Vec4i> hierarchy, const char* title, Mat imgref){
    Mat drawing = Mat::zeros( imgref.size(), CV_8UC3 );
    //Mat drawing = src;
    for( size_t i = 0; i< contoursi.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contoursi, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
    imshow( title, drawing );
}

void showContoursAndCenters(std::vector<std::vector<Point> > contoursi, std::vector<Vec4i> hierarchy,
    std::vector<Point2f> centers, const char* title, Mat imgref){
    Mat drawing = Mat::zeros( imgref.size(), CV_8UC3 );
    //Mat drawing = src;
    for( size_t i = 0; i < contoursi.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contoursi, (int)i, color, 2, LINE_8, hierarchy, 0 );
        circle( drawing, Point( floor(centers[i].x), floor(centers[i].y) ) , 5, color );
    }

    imshow( title, drawing );
}


void showContoursAndCentersOnImage(std::vector<std::vector<Point> > contoursi, std::vector<Vec4i> hierarchy,
    std::vector<Point2f> centers, const char* title, Mat imgref){
    Mat drawing = imgref;
    //Mat drawing = src;
    for( size_t i = 0; i < contoursi.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contoursi, (int)i, color, 2, LINE_8, hierarchy, 0 );
        circle( drawing, Point( floor(centers[i].x), floor(centers[i].y) ) , 5, color );
    }

    imshow( title, drawing );
}


void showEncodingInformation(  std::vector<int> code, int segment_length, int encoding_length, const char* title, Mat imgref ){
    if( code.size() != encoding_length*encoding_length ){
        std::cout << "[ERROR]  ENCODING LENGTH DOESNT MATCH CODE SIZE CANNOT PLOT GRID SHOWCASE" << std::endl;
    }
    Mat drawing = imgref;
    for( int ii = 1; ii < encoding_length; ii++ ){
        line( drawing, Point( ii*segment_length ,0), Point( ii*segment_length ,imgref.size().height), Scalar( 0,0,255 ), 3 );
    }

    for( int ii = 1; ii < encoding_length; ii++ ){
        line( drawing, Point( 0, ii*segment_length ), Point( imgref.size().width, ii*segment_length), Scalar( 0,0,255 ), 3 );
    }

    double shift = 0.2;
    for( int ii = 0; ii < encoding_length; ii++ ){
        for( int jj = 0; jj < encoding_length; jj++ ){
            int codei = code[jj*encoding_length + ii];
            String txt;
            if( codei == -1 ) txt = "X";
            else txt = std::to_string( codei );
            Point txt_point = Point( cvRound(ii*segment_length + (0.5-shift)*segment_length) , cvRound(jj*segment_length + (0.5+shift)*segment_length) );
            putText( drawing, txt, txt_point ,FONT_HERSHEY_PLAIN, 1, Scalar( 0,0,255 ),2 );
        }
    }
    
    imshow( title, drawing );
}