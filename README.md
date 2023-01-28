# [Spring23] Lift-CDUS project
---
This repository implements the computer vision stack for the Lift-CDUS project.

# Quick Start

Download the repository and create the `conda` environment. Note that this has 
to be done only once.

```
git clone --recurse-submodules REPOSITORY <local_name>
cd <local_name>
conda env create --name <environment_name> --file environment.yaml
```

Every time you want to use this software you'll have to execute the following 
commands:

Activate the `conda` environment 

```
conda activate <environment_name>
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

## Second
I also trained the `yolo` neural network to detect a custom made feature (jack 
sparrow). I tested this with the `Intel Realsense D435`.


