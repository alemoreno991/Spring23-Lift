SRC_DIR := $(PWD)/src
YOLO_DIR := ${SRC_DIR}/yolov7
CUSTOM_YOLO_DIR = $(PWD)/customize_yolo
export PYTHONPATH := ${YOLO_DIR}

# Training configuration
EPOCHS = 500
IMG_SIZE = 680 680
PROJ_DIR = $(CUSTOM_YOLO_DIR)/runs/train
DATA_CONFIG = $(CUSTOM_YOLO_DIR)/data/custom_data.yaml 
HYP_CONFIG = $(YOLO_DIR)/data/hyp.scratch.custom.yaml
CFG_CONFIG = $(CUSTOM_YOLO_DIR)/cfg/training/custom_cfg.yaml
NAME = yolov7-tiny-custom
WEIGHTS = $(CUSTOM_YOLO_DIR)/runs/default/yolov7/weights/default/yolov7-tiny.pt

# Run the program configuration 
MAIN = main.py
WEIGHTS_CUSTOM = $(PWD)/customize_yolo/runs/train/$(NAME)/weights/best.pt

run_default:
	cd $(SRC_DIR); python $(MAIN) --conf 0.8 

run_custom:
	cd $(SRC_DIR); python $(MAIN) --weights $(WEIGHTS_CUSTOM)

test_on_video:
	cd $(YOLO_DIR) && \
	python detect.py --weights $(WEIGHTS_CUSTOM) \
		--source $(PWD)/Pickle_skydio_abridged_720.mp4 \
		--project $(CUSTOM_YOLO_DIR)/runs/detect

train:
	cd $(YOLO_DIR) && \
	python train.py --workers 1 --device 0 --batch-size 2 \
	--epochs $(EPOCHS) --img $(IMG_SIZE) --data $(DATA_CONFIG) --hyp $(HYP_CONFIG) \
	--cfg $(CFG_CONFIG) --project $(PROJ_DIR) --name $(NAME) --weights $(WEIGHTS)
