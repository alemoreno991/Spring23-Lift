# Spring23-Lift
---
This repository implements the computer vision stack for the Lift-CDUS project.

# What did I do?

## First
I implemented a image detector based on `yoloV7` that tells you the 3D location,
relative to the camera, of the detected feature. I tested it with the Realsense
D435 and it worked.

## Second
I also trained the yolo neural network to detect a custom made feature (jack 
sparrow). I tested this with the Realsense D435.

# Quick start

- activate the conda environment 
- go to src/yolov7
- connect the `Intel Realsense` Camera (I'm using the D435)
- execute `python main.py`

# Requirements

- conda
- yolov7
- realsense


