export INPUT_JSONL_FILE="../../output/processed_data/data_stage2.jsonl"
export OUTPUT_DIR="../../input/videos_split_fps24"

cd Practical-RIFE
python inference_video_list.py --filelist_jsonl $INPUT_JSONL_FILE --output_dir $OUTPUT_DIR --fp16 --fps 24
