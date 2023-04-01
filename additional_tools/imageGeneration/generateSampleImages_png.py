#!/usr/bin/env python3
import cv2 as cv
import numpy as np

PX2IN_SCALE = 40


def createImage8bitEncoding(num):
    if num > 256 or num < 0: return -1
    bin_num_string = bin(num)[2:]
    be8 = [0,0,0,0,0,0,0,0]
    for ii in range(0, len(bin_num_string)):
        be8[ 8 - len(bin_num_string) + ii ] = int(bin_num_string[ii])
    print(be8)
    return be8




def drawIdentifiers():
    # CREATE EMPTY CANVAS
    drawing = 255*np.ones( ( 18*PX2IN_SCALE, 18*PX2IN_SCALE, 3 ), np.uint8 )

    # DRAW RECTANGULAR IDENTIFIERS
    drawing = cv.rectangle( drawing, \
                            ( int(1.5*PX2IN_SCALE), int(4.5*PX2IN_SCALE) ), \
                            ( int(3*PX2IN_SCALE), int(13.5*PX2IN_SCALE) ), \
                            ( 0, 0, 0), \
                            cv.FILLED \
                        )

    drawing = cv.rectangle( drawing, \
                            ( int(4.5*PX2IN_SCALE), int(1.5*PX2IN_SCALE) ), \
                            ( int(13.5*PX2IN_SCALE), int(3*PX2IN_SCALE) ), \
                            ( 0, 0, 0), \
                            cv.FILLED \
                        )

    drawing = cv.rectangle( drawing, \
                            ( int(4.5*PX2IN_SCALE), int(15*PX2IN_SCALE) ), \
                            ( int(13.5*PX2IN_SCALE), int(16.5*PX2IN_SCALE) ), 
                            ( 0, 0, 0), \
                            cv.FILLED \
                        )

    drawing = cv.rectangle( drawing, \
                            ( int(15*PX2IN_SCALE), int(4.5*PX2IN_SCALE) ), \
                            ( int(16.5*PX2IN_SCALE), int(13.5*PX2IN_SCALE) ), \
                            ( 0, 0, 0), \
                            cv.FILLED \
                        )
    
    # DRAW CIRCULAR IDENTIFIER
    drawing = cv.circle( drawing, \
                         ( int(6*PX2IN_SCALE), int(6*PX2IN_SCALE) ), \
                         int(1.5*PX2IN_SCALE), \
                         ( 0, 0, 0), \
                         cv.FILLED \
                        )

    drawing = cv.circle( drawing, \
                         ( int(6*PX2IN_SCALE), int(6*PX2IN_SCALE) ), \
                         int(1*PX2IN_SCALE), \
                         ( 255, 255, 255), \
                         cv.FILLED \
                        )

    drawing = cv.circle( drawing, \
                         ( int(6*PX2IN_SCALE), int(6*PX2IN_SCALE) ), \
                         int(0.5*PX2IN_SCALE), \
                         ( 0, 0, 0), \
                         cv.FILLED \
                        )

    return drawing



def drawEncoding( encoding_list, drawing ):
    encoded_drawing = drawing
    for ii in range( 0, 3 ):
        for jj in range( 0, 3 ):
            if ii == 0 and jj == 0: 
                continue
            if encoding_list[ (ii * 3 + jj) - 1 ] == 1:
                continue
            else:
                encoded_drawing = cv.rectangle( encoded_drawing, \
                                                ( int( (4.5 + 3*jj) * PX2IN_SCALE), int( (4.5 + 3*ii) * PX2IN_SCALE) ), \
                                                ( int( (7.5 + 3*jj) * PX2IN_SCALE), int( (7.5 + 3*ii) * PX2IN_SCALE) ), \
                                                ( 0, 0, 0), \
                                                cv.FILLED \
                                            )
    return encoded_drawing




def main():
    for ii in range(0,10):
        filenamei =  "./png_images/" + str(ii) + ".png"
        img_w_idbars = drawIdentifiers()
        bitencoding = createImage8bitEncoding(ii)
        encoded_img = drawEncoding(bitencoding, img_w_idbars)
        cv.imwrite(filenamei, encoded_img)


if __name__ == '__main__':
    main()