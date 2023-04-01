
# SAMPLE IMAGE GENERATORS
## 1. DEPENDENCIES
#
### 1a. SVG
The svg sample image generator (generateSampleImages_svg.py) is dependent on [drawsvg](https://github.com/cduck/drawsvg) and the library can be installed with pip with the following command:

    python3 -m pip install "drawsvg~=2.0"

### 1b. PNG
The png sample image generator (generateSampleImages_png.py) is dependent on [opencv](https://opencv.org) whose installation instructinos can be found through the link
#
## 2. GENERATING IMAGES
#
### 2a. SVG
Using the generateSampleImages_svg.py script with the following command will generate all 255 variations and export each as an svg image deposited into the svg_images folder:

    python3 generateSampleImages_svg.py

### 2b. PNG
Using the generateSampleImages_png.py with the following command will generate all 255 variations and export each as an svg image deposited into the png_images folder:

    python3 generateSampleImages_png.py