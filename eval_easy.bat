@echo off
yolo task=detect mode=val model=yolov8nbest.pt data=easy.yaml split=test
pause