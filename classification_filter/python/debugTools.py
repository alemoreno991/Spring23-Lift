import math 
import random as rnd
import cv2 as cv
import numpy as np

def showContours( contoursi, hierarchy , title, imageref ):
    drawing = np.zeros( imageref.shape , dtype=np.uint8)
    for ii in range(0,len(contoursi)): 
        color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
        cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
    cv.imshow( title, drawing )

def showContoursAndCenters( contoursi, hierarchy , centers, title, imageref ):
    drawing = np.zeros( imageref.shape , dtype=np.uint8)
    for ii in range(0,len(contoursi)): 
        color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
        cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
        cv.circle( drawing, ( int(centers[ii][0]) , int(centers[ii][1]) ) , 5, color )
    cv.imshow( title, drawing ) 

def showContoursAndCentersOnImage( contoursi, hierarchy , centers, title, imageref ):
    drawing = imageref
    for ii in range(0,len(contoursi)): 
        color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
        cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
        cv.circle( drawing, ( int(centers[ii][0]) , int(centers[ii][1]) ) , 5, color )
    cv.imshow( title, drawing )  

def showEncodingInformation( code, segment_length, encoding_length, title, imgref ):
    drawing = imgref
    red_color = (int(0),int(0),int(255))
    img_height, img_width, _ = imgref.shape

    for ii in range(1,encoding_length):
        cv.line( drawing, ( int(ii*segment_length) , int(0) ), ( ii*segment_length ,img_height), red_color, 3  )

    for ii in range(1,encoding_length):
        cv.line( drawing, ( int(0), int(ii*segment_length) ), ( img_width, ii*segment_length), red_color, 3  )

    shift = 0.2
    for ii in range(0,encoding_length):
        for jj in range(0,encoding_length):
            codei = code[jj*encoding_length + ii]
            txt = ''
            if( codei == -1 ):
                txt = "X"
            else: 
                txt = str( codei )
            txt_point = ( int(ii*segment_length + (0.5-shift)*segment_length) , int(jj*segment_length + (0.5+shift)*segment_length) )
            cv.putText( drawing, txt, txt_point , cv.FONT_HERSHEY_PLAIN, 1, red_color , 2 )

    cv.imshow(title,drawing)