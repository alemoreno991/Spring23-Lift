#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

using namespace cv;

// ASSITISTING FUNCTIONS - START
double euclideanDistance( Point a , Point b ){
    return std::pow( double( std::pow(a.x - b.x,2) + std::pow(a.y - b.y,2) ) , 0.5);
};

double angleBetweenPoints( Point a , Point b ){
    double angle = std::atan2( double(b.y - a.y) , double(b.x - a.x) );
    if( angle < 0 ) angle = 2*M_PI + angle;
    return angle;
};

int determineAngleQuadrant( double angle ){
    // DONT NEED TO CHECK LOWER BECAUSE ANGLES EXIST IN 0,2 PI]
    if( angle < M_PI/2  ) return 0; 
    else if( angle < M_PI ) return 1;
    else if( angle < 3.*M_PI/2. ) return 2;
    return 3;
}
// ASSITISTING FUNCTIONS - END 