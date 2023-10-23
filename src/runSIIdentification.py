import argparse

# IDENTIFICATION STACK 
import identification_stack.identificationStack as idstack

# OPENCV
import cv2 as cv

if __name__ == '__main__':

    stack_opts                  = 0  # SINGLE IMAGE IDENTIFICATION
    id_stack                    = idstack.identificationStack( stack_opts )

    # # EXAMPLE USAGE
    # test_img1                   = cv.imread("test1_temp.png")
    # assert test_img1 is not None, 'Image Not Found '
    # read_code, crate_center_pt  = id_stack.runIdentification( test_img1 )
    # print(read_code)
    # print("\n")
    # print(crate_center_pt)
    # image_check = cv.circle( test_img1.copy(), (int(crate_center_pt[0]),int(crate_center_pt[1])), 20, (int(255), int(0), int(0)), 3)
    # cv.imwrite("check_1.png",image_check)

    # test_img2                   = cv.imread("test2_temp.png")
    # assert test_img1 is not None, 'Image Not Found '
    # read_code, crate_center_pt  = id_stack.runIdentification( test_img2 )
    # print(read_code)
    # print("\n")
    # print(crate_center_pt)
    # image_check = cv.circle( test_img1.copy(), (int(crate_center_pt[0]),int(crate_center_pt[1])), 20, (int(255), int(0), int(0)), 3)
    # cv.imwrite("check_2.png",image_check)
