# CLASSIFICATION FILTER
## BUILDING

First create a bin and build folder with the following commands (if they are yet to exist already):

    [PROJECT_DIR] mkdir build bin

Build this project of the repository using cmake: 

    [PROJECT_DIR] cd build 
    [PROJECT_DIR] cmake ..
    [PROJECT_DIR] make

After building, you should find the <strong>decodeImage</strong> binary inside the bin folder created earlier
#
## TESTING IMAGES
Using the <strong>decodeImage</strong>  binary, you can decode images of the crate marker with the following:

    [PROJECT_DIR] ./bin/decodeImage <PATH-TO-IMAGE>

#
## DEBUG AND SHOWCASE MODES
The following macro determines your debug mode: <strong>DECODER_DEBUG_MODE </strong>

* 0 - NONE
* 1 - DEBUGGER MODE FOR D1 EXTRACTION
* 2 - DEBUGGER MODE FOR ENCODING DETERMINATION
* 3 - DEBUGGER MODE FOR BOTH MODES (1)(2)


The following macro determines your showcase mode: <strong>DECODER_SHOWCASE_MODE </strong>

* 0 - NONE
* 1 - PRESENTS DECODING STEPS IN A PRESENTABLE FASHION