SRC := $(PWD)/src
export PYTHONPATH := ${SRC}/yolov7

all:
	cd $(SRC); python main.py
