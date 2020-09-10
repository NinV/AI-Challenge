#!/usr/bin/env bash
echo "Start to process AICity HCM :"
starttime=`date +%s`
echo "---------------------------------------------------"
#get all videos names
ROOT_PATH=$(pwd)
python3 detect_and_track_final_submission.py --input /data/test_data --weights detection/yolov5/weights/yolov5x.pt
cd counting
python3 counting.py
python3 submit.py
mv submit.txt /data/submission_output
