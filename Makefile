SRC_DIR := $(PWD)/src
YOLO_DIR := ${SRC_DIR}/yolov7
CUSTOM_YOLO_DIR = $(PWD)/customize_yolo
export PYTHONPATH := ${YOLO_DIR}

# Run the program configuration
MAIN = main.py

# Training configuration
EPOCHS = 100
IMG_SIZE = "640 640"
DATA_CONFIG = $(CUSTOM_YOLO_DIR)/data/custom_data.yaml 
HYP_CONFIG = $(YOLO_DIR)/data/hyp.scratch.custom.yaml
CFG_CONFIG = $(CUSTOM_YOLO_DIR)/cfg/training/custom_cfg.yaml
NAME = yolov7-custom
WEIGHTS = $(CUSTOM_YOLO_DIR)/weights/yolov7.pt

run:
	cd $(SRC); python $(MAIN)

train:
	cd $(YOLO_DIR) && \
	python train.py --workers 1 --device 0 --batch-size 8 \
	--epochs $(EPOCHS) --img $(IMG_SIZE) --data $(DATA_CONFIG) --hyp $(HYP_CONFIG) \
	--cfg $(CFG_CONFIG) --name $(NAME) --weights $(WEIGHTS)
