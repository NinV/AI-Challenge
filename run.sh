#!/usr/bin/env bash
echo "Start to process AICity HCM :"
starttime=`date +%s`
echo "---------------------------------------------------"
#get all videos names
ROOT_PATH=$(pwd)
python3 detect_and_track_final_submission.py --input /data/test_data
cd counting
python counting.py
python submit.py
mv submit.txt /data/submission_output
