import math 
import random as rnd
import cv2 as cv
import numpy as np

# TODO: CLEAN THIS SECTION UP

def showContours( contoursi , title:str , imageref_shape ):
    drawing = np.zeros( imageref_shape , dtype=np.uint8)
    if (len(contoursi) > 0):
        for ii in range(0,len(contoursi)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
    cv.imshow( title, drawing )



def showContoursOnImage( contoursi , title:str, imageref ):
    drawing = imageref
    if (len(contoursi) > 0):
        for ii in range(0,len(contoursi)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
    cv.imshow( title, drawing )

def drawContoursOnImage( contoursi , imageref ):
    drawing = imageref
    if (len(contoursi) > 0):
        for ii in range(0,len(contoursi)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
    return drawing

def showContoursAndAreas(contours , title:str , imageref_shape ):
    drawing = np.zeros( imageref_shape , dtype=np.uint8)
    if (len(contours) > 0):
        for ii in range(0,len(contours)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            
            _, rectangle_size, _ = cv.minAreaRect(contours[ii])
            rect_area = rectangle_size[0] * rectangle_size[1]
            rect_area = round(rect_area, 2)


            approx_cp = cv.approxPolyDP( contours[ii], 0.025*cv.arcLength(contours[ii],True), True )
            approx_cp_area = abs( cv.contourArea(approx_cp) )


            txt = str(ii) + " RA:"+str(rect_area) + " PA:" + str(approx_cp_area)
            txt_point = tuple(contours[ii][0][0])
            txt_point = (txt_point[0]-20, txt_point[1]-10)

            cv.putText( drawing, txt, tuple(txt_point) , cv.FONT_HERSHEY_PLAIN, 1, color , 2 )
            cv.drawContours( drawing, [contours[ii]], 0, color, 3 )

    cv.imshow( title, drawing )


def showContoursAndCenters( contoursi , centers, title:str, imageref ):
    drawing = np.zeros( imageref.shape , dtype=np.uint8)
    if (len(contoursi) > 0):
        for ii in range(0,len(contoursi)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
            cv.circle( drawing, ( int(centers[ii][0]) , int(centers[ii][1]) ) , 5, color )
    cv.imshow( title, drawing ) 

def showContoursAndCentersOnImage( contoursi , centers, title:str, imageref ):
    drawing = imageref
    if (len(contoursi) > 0):
        for ii in range(0,len(contoursi)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            cv.drawContours( drawing, [contoursi[ii]], 0, color, 3 )
            cv.circle( drawing, ( int(centers[ii][0]) , int(centers[ii][1]) ) , 5, color )
    cv.imshow( title, drawing )  

def showContoursAndPerimeters(contours , title:str , imageref_shape ):
    drawing = np.zeros( imageref_shape , dtype=np.uint8)
    if (len(contours) > 0):
        for ii in range(0,len(contours)): 
            color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
            
            contour_perimeter = cv.arcLength(contours[ii],True)
            contour_perimeter = round(contour_perimeter, 4)
            txt = str(ii) + " " + str(contour_perimeter)
            txt_point = tuple(contours[ii][0][0])
            txt_point = (txt_point[0]-20, txt_point[1]-10)
            cv.putText( drawing, txt, tuple(txt_point) , cv.FONT_HERSHEY_PLAIN, 1, color , 2 )
            cv.drawContours( drawing, [contours[ii]], 0, color, 3 )

    cv.imshow( title, drawing )

def showEncodingInformation( code, segment_length, encoding_length, title:str, imgref ):
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

def showAxesOnImage(rvec, tvec, CAMERA_MATRIX, DISTANCE_COEFFICIENTS, title:str, imgref):
    drawing = imgref
    axis_length = imgref.shape[0]/5
    axis_points, _ = cv.projectPoints(np.float32([[0, 0, 0], 
                                                [axis_length, 0, 0],
                                                [0, axis_length, 0], [0, 0, axis_length]]),
                                                rvec, tvec, CAMERA_MATRIX, DISTANCE_COEFFICIENTS)
    cv.line(drawing, tuple(axis_points[0].astype(int).ravel()), tuple(axis_points[1].astype(int).ravel()), (0, 0, 255), 3)  # X-axis (red)
    cv.line(drawing, tuple(axis_points[0].astype(int).ravel()), tuple(axis_points[2].astype(int).ravel()), (0, 255, 0), 3)  # Y-axis (green)
    cv.line(drawing, tuple(axis_points[0].astype(int).ravel()), tuple(axis_points[3].astype(int).ravel()), (255, 0, 0), 3)  # Z-axis (blue)
    cv.imshow(title, drawing)

def showPointsOnImage(points,title:str, image):
    drawing = image
    for ii in range (0, len(points)):
        color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
        cv.circle( drawing, (int(points[ii][0]) , int(points[ii][1])) , 10, color, 3 )
    cv.imshow( title, drawing )

def showContoursAndBoundingRect( contours, rect, title:str, imageref ):
    drawing = np.zeros( imageref, dtype=np.uint8)
    center,size,_ = rect
    rect_vertices = cv.boxPoints(rect)
    rect_vertices = np.int0(rect_vertices)
    # Draw the bounding rectangle on the image
    cv.polylines(drawing, [rect_vertices], True, (255, 255, 0), 2)
    for ii in range(0,len(contours)): 
        color = ( int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)), int(rnd.randrange(0,255,1)) )
        cv.drawContours( drawing, [contours[ii]], 0, color, 3 )
    cv.imshow( title, drawing ) 

