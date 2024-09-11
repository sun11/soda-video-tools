export VIDEO_FILELIST="config/data.jsonl"
export OUTPUT_FILELIST="../output/processed_data/data_stage1.jsonl"
export OUTPUT_FOLDER="../input/videos_split"

python pyscenedetect_vcut.py \
    $VIDEO_FILELIST \
    --threshold 27 \
    --frame_skip 0 \
    --min_seconds 2 \
    --max_seconds 5 \
    --save_dir $OUTPUT_FOLDER \
    --num_processes 12
python write_jsonl_list.py $OUTPUT_FOLDER $OUTPUT_FILELIST