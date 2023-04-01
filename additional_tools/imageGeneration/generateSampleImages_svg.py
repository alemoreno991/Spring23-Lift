#!/usr/bin/env python3
import drawsvg as dw

PX2IN_SCALE = 40


def createImage8bitEncoding(num):
    if num > 256 or num < 0: return -1
    bin_num_string = bin(num)[2:]
    be8 = [0,0,0,0,0,0,0,0]
    for ii in range(0, len(bin_num_string)):
        be8[ 8 - len(bin_num_string) + ii ] = int(bin_num_string[ii])
    print(be8)
    return be8




def drawIdentifiers(id):
    # CREATE EMPTY CANVAS
    drawing = dw.Drawing( 18*PX2IN_SCALE, 18*PX2IN_SCALE, id_prefix=id)
    drawing.append( \
        dw.Rectangle( int(0), int(0), int(18*PX2IN_SCALE), int(18*PX2IN_SCALE), fill='white' ) \
    )    

    # DRAWING RECTANGULAR IDENFIER
    drawing.append( \
        dw.Rectangle( int(1.5*PX2IN_SCALE), int(4.5*PX2IN_SCALE), int(1.5*PX2IN_SCALE), int(9*PX2IN_SCALE) ) \
    )
    drawing.append( \
        dw.Rectangle( int(4.5*PX2IN_SCALE), int(1.5*PX2IN_SCALE), int(9*PX2IN_SCALE), int(1.5*PX2IN_SCALE) ) \
    )
    drawing.append( \
        dw.Rectangle( int(4.5*PX2IN_SCALE), int(15*PX2IN_SCALE) , int(9*PX2IN_SCALE), int(1.5*PX2IN_SCALE) ) \
    )
    drawing.append( \
        dw.Rectangle( int(15*PX2IN_SCALE), int(4.5*PX2IN_SCALE) , int(1.5*PX2IN_SCALE), int(9*PX2IN_SCALE) ) \
    )
    
    # DRAW CIRCULAR IDENTIFIER
    drawing.append( \
        dw.Circle( int(6*PX2IN_SCALE), int(6*PX2IN_SCALE), int(1.5*PX2IN_SCALE) ) \
    )
    drawing.append( \
        dw.Circle( int(6*PX2IN_SCALE), int(6*PX2IN_SCALE), int(1*PX2IN_SCALE), fill = 'white') \
    )
    drawing.append( \
        dw.Circle( int(6*PX2IN_SCALE), int(6*PX2IN_SCALE), int(0.5*PX2IN_SCALE)) \
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
                encoded_drawing.append( \
                    dw.Rectangle( int( (4.5 + 3*jj) * PX2IN_SCALE), \
                                  int( (4.5 + 3*ii) * PX2IN_SCALE), \
                                  int(3*PX2IN_SCALE), int(3*PX2IN_SCALE) \
                    )\
                )
    return encoded_drawing




def main():
    print("GENERATING IMAGES...")
    for ii in range(0,255):
        filenamei =  "./svg_images/" + str(ii) + ".svg"
        idi = "code_" + str(ii)
        img_w_idbars = drawIdentifiers(idi)
        bitencoding = createImage8bitEncoding(ii)
        encoded_img = drawEncoding(bitencoding, img_w_idbars)
        encoded_img.save_svg(filenamei)
    print("DONE")


if __name__ == '__main__':
    main()