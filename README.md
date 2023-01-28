# [Spring23] Lift-CDUS project
---
This repository implements the computer vision stack for the Lift-CDUS project.

# Quick Start

Download the repository and create the `conda` environment. Note that this has 
to be done only once.

```
git clone --recurse-submodules REPOSITORY <local_name>
cd <local_name>
conda env create --file environment.yml
```

Every time you want to use this software you'll have to execute the following 
commands:

Activate the `conda` environment 

```
conda activate CDUS-Lift
```

Run the program (make sure the Intel Realsense D435 is connected)

```
make 
```

# What did I do?

## First
I implemented a image detector based on `yoloV7` that tells you the 3D location,
relative to the camera, of the detected feature. I tested it with the `Intel 
Realsense D435` and it worked.

There are three important files: 
- `main.py`: it integrates the camera and the detector to localize the feature 
in the 3D world. 
- `yolo.py`: it abstracts the yolo-based detector 
- `realsense.py`: it abstracts the camera stuff 

## Second
I also trained the `yolo` neural network to detect a custom made feature (jack 
sparrow). I tested this with the `Intel Realsense D435`.

This is the step-by-step procedure:

- Obtain the photos of the feature you want `yolo` to detect
- Use the open-source [labelImg](https://github.com/heartexlabs/labelImg) 
software to put a labeled box around the feature in each photo. Make sure to use
`yolo` format.
- Augment (rotations, scaling, brightness, blur, etc) the custom dataset 
(I didn't do it but we should.)
- Separate a training (\~80%) and validation (\~20%) set.
```
    --- /train
          |
          |---> /images
          |---> /labels
    
    --- /val
          |
          |---> /images
          |---> /labels
```
- Copy the `train` and `val` folders to `./customize_yolo/data`
- Modify the `customize_yolo/data/custom_data.yaml` file to match your project 
- Modify the `customize_yolo/cfg/training/custom_cfg.yaml` file to match your project 

Finally, go back to the `root` and execute the following command 

```
make train
```

# References

Understanding `yolo` and `Realsense`

- [How To Deploy YOLOv7 on Live Webcam for Custom Object Detection with OpenCV](https://www.youtube.com/watch?v=XzUMigbYRUI&t=452s)
- [Object detection with YOLO V3 and Realsense camera](https://www.youtube.com/watch?v=6Ps7oOqoJaw&t=376s)
- [How To Use YOLOv7 Model for Object Detection](https://www.youtube.com/watch?v=IboFrLHwxDg&t=10s)

Understanding 3D positioning

- [Position estimation of an object with YOLO using RealSense](https://www.youtube.com/watch?v=--81OoXMvlw&t=260s)

Understanding `yolo` customization

- [Official YOLO v7 Custom Object Detection Tutorial | Windows & Linux](https://www.youtube.com/watch?v=-QWxJ0j9EY8&t=1s)
- [Official YOLOv7 | Object Detection](https://www.youtube.com/watch?v=n0Lp59zjQPE&t=2s)
- [Albumentations](https://albumentations.ai/docs/)
- [Albumentation tutorial](https://www.youtube.com/watch?v=rAdLwKJBvPM&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=12&t=39s)
