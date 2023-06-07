# [Spring23] Lift-CDUS project
---

The idea is to develop the code in the corresponding branches:

- detector-dev
- classifier-dev

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

Run the program in its default configuration (make sure the Intel Realsense D435 
is connected)

```
make run_default
```

Or run it with your customized neural network

```
make run_custom
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

In order to use the custom `yolo` network use the following command

```
make run_custom
```

## Third

A `Unity` simulation of the **J.J.Pickle Research Center** was implemented. 
Then, the `Unity Perception Package` was used to generate a randomized synthetic
dataset (lighting, rotations, translations, color, hue). 

Since the output format offered by `Unity` is not compatible with `YOLOv7`, 
[Roboflow](https://app.roboflow.com/) was used to further augment the dataset:

![Roboflow Augmentation](./doc/img/roboflow_augmentation.png)

Finally, the dataset was exported in `yoloV7` format and trained locally as 
explained in the previous section. The dataset has ~2.5K images and was split 
into train, validation and test (~80%, ~15%, ~5%).

# Data

Since GitHub does not offer a lot of storage, the data used during this project
can be found in the [UT Box](https://utexas.app.box.com/folder/193679796828) 
in the `CDUS-development/raw_data` folder.

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

Unity Perception

- [Unity Perception Tutorial](https://www.youtube.com/watch?v=mkVE2Yhe454&t=1251s)
- [Generate a custom marker](https://www.youtube.com/watch?v=om6QtMb_wwo&t=146s)
- [Simulating J.J.Pickle](https://www.youtube.com/watch?v=ddy12WHqt-M&t=492s)

Roboflow

- 
