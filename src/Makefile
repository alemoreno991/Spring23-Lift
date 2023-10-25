SRC_DIR := $(PWD)/identification_stack/marker_identification/detection_nnetwork/src
YOLO_DIR := ${SRC_DIR}/yolov7
CUSTOM_YOLO_DIR = $(PWD)/detection_nnetwork/customize_yolo
export PYTHONPATH := ${YOLO_DIR}:${PYTHONPATH}

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
WEIGHTS_CUSTOM = $(PWD)/identification_stack/detection_nnetwork/customize_yolo/runs/train/$(NAME)/weights/best.pt

train:
	cd $(YOLO_DIR) && \
	python train.py --workers 1 --device 0 --batch-size 2 \
	--epochs $(EPOCHS) --img $(IMG_SIZE) --data $(DATA_CONFIG) --hyp $(HYP_CONFIG) \
	--cfg $(CFG_CONFIG) --project $(PROJ_DIR) --name $(NAME) --weights $(WEIGHTS)

run_sitest_identification:
	python3 runSIIdentification.py

compile_veronte_comm:
	g++ -I./include veronteCommServer.cpp -o tele_server -pthread -lzmq