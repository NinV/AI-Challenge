#!/usr/bin/env bash
echo "Start to process AICity HCM :"
starttime=`date +%s`
echo "---------------------------------------------------"
#get all videos names
ROOT_PATH=$(pwd)

# default batch_size = 12 for 2070 Super
python3 detect_and_track_final_submission.py --input /data/test_data  --output counting/track_results --weights detection/yolov5/weights/yolov5x.pt --batch_size 12

# for 1080TI batch_size can be set to higher value
# python3 detect_and_track_final_submission.py --input /data/test_data --output counting/track_results --weights detection/yolov5/weights/yolov5x.pt --batch_size 20

cd counting
python3 counting.py
python3 submit.py
mv submit.txt /data/submission_output
