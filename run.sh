#!/usr/bin/env bash
echo "Start to process AICity HCM :"
starttime=`date +%s`
echo "---------------------------------------------------"
#get all videos names
ROOT_PATH=$(pwd)
source ~/anaconda3/bin/activate
conda activate yolo
python detect_and_track_final_submission.py --input /home/dungpv/Documents/videos/ --output counting/track_results
cd counting
python counting.py
python submit.py