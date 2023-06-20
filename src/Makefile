SRC_DIR := $(PWD)/detection_nnetwork/src
YOLO_DIR := ${SRC_DIR}/yolov7
CUSTOM_YOLO_DIR = $(PWD)/detection_nnetwork/customize_yolo
export PYTHONPATH := ${YOLO_DIR}

# Training configuration
EPOCHS = 30
IMG_SIZE = 800 800
PROJ_DIR = $(CUSTOM_YOLO_DIR)/runs/train
DATA_CONFIG = $(CUSTOM_YOLO_DIR)/data/custom_data.yaml 
HYP_CONFIG = $(YOLO_DIR)/data/hyp.scratch.custom.yaml
CFG_CONFIG = $(CUSTOM_YOLO_DIR)/cfg/training/custom_cfg.yaml
WEIGHTS = $(CUSTOM_YOLO_DIR)/runs/default/yolov7/weights/default/yolov7-tiny.pt

# Run the program configuration 
MAIN = main.py
NAME = yolov7-tiny-custom
WEIGHTS_CUSTOM = $(PWD)/detection_nnetwork/customize_yolo/runs/train/$(NAME)/weights/best.pt

# TEST_VIDEO = $(PWD)/test_items/videos/test06132023.mp4
TEST_ITEM = $(PWD)/test_items/images/test06132023.png

run_default:
	cd $(SRC_DIR); python $(MAIN) --conf 0.8 

run_custom:
	cd $(SRC_DIR); python $(MAIN) --weights $(WEIGHTS_CUSTOM)

test_on_item:
	cd $(YOLO_DIR) && \
	python3 detect.py --weights $(WEIGHTS_CUSTOM) \
		--source $(TEST_ITEM) \
		--conf 0.75 \
		--project $(CUSTOM_YOLO_DIR)/runs/detect

test_composite_on_item:
	python3 detectc.py --weights $(WEIGHTS_CUSTOM) \
		--source $(TEST_ITEM) \
		--conf 0.75 \
		--project $(CUSTOM_YOLO_DIR)/runs/detect

train:
	cd $(YOLO_DIR) && \
	python train.py --workers 1 --device 0 --batch-size 2 \
	--epochs $(EPOCHS) --img $(IMG_SIZE) --data $(DATA_CONFIG) --hyp $(HYP_CONFIG) \
	--cfg $(CFG_CONFIG) --project $(PROJ_DIR) --name $(NAME) --weights $(WEIGHTS)
