export INPUT_LIST_PATH="../output/processed_data/data_stage1.jsonl"
export RESOLUTION_SAVE_PATH="../output/processed_data/filtered_resolution.jsonl"
export TEXT_SCORE_SAVE_PATH="../output/processed_data/text_score.jsonl"
export MOTION_SCORE_SAVE_PATH="../output/processed_data/motion_score.jsonl"
export FILTER_BY_TEXT_SAVE_PATH="../output/processed_data/filtered_text.jsonl"
export VQA_CSV_FILE="../output/processed_data/videos_quality_score.csv"
export VQA_JSONL_FILE="../output/processed_data/video_quality_score.jsonl"
export OUTPUT_STAGE2_JSONL_FILE="../output/processed_data/data_stage2.jsonl"

python filter_videos_by_resolution.py $INPUT_LIST_PATH $RESOLUTION_SAVE_PATH

# Get text score of all videos
accelerate launch compute_text_score.py \
    --input_jsonl=$RESOLUTION_SAVE_PATH \
    --video_path_column="video_path" \
    --saved_freq=10 \
    --saved_path=$TEXT_SCORE_SAVE_PATH

python filter_videos_by_text_score.py $TEXT_SCORE_SAVE_PATH $FILTER_BY_TEXT_SAVE_PATH

# Get motion score after filter videos by asethetic score and text score
python compute_motion_score.py \
    --video_metadata_path=$TEXT_SCORE_SAVE_PATH \
    --video_path_column="video_path" \
    --saved_freq=10 \
    --saved_path=$MOTION_SCORE_SAVE_PATH \
    --n_jobs=8 \
    --text_score_metadata_path $TEXT_SCORE_SAVE_PATH

python compute_video_quality_score.py --input_jsonl $FILTER_BY_TEXT_SAVE_PATH --output_result_csv $VQA_CSV_FILE --output_jsonl $VQA_JSONL_FILE

python coarse_filter.py --input_jsonl $FILTER_BY_TEXT_SAVE_PATH \
    --text_score $TEXT_SCORE_SAVE_PATH \
    --motion_score $MOTION_SCORE_SAVE_PATH \
    --vqa_score $VQA_JSONL_FILE \
    --output_jsonl $OUTPUT_STAGE2_JSONL_FILE \
    --top_k 100000

dj-process --config config/tagging.yaml