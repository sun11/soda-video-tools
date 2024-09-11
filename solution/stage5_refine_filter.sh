export TAG_JSONL_FILE="../output/processed_data/data_stage1_tagging.jsonl"
export CAPTION_JSONL_FILE="../output/processed_data/data_stage3.jsonl"
export VQA_JSONL_FILE="../output/processed_data/video_quality_score.jsonl"
export MOTION_SCORE_JSONL_FILE="../output/processed_data/motion_score.jsonl"
export OUTPUT_SCORE_JSONL_FILE="../output/processed_data/final_score.csv"
export OUTPUT_JSONL_FILE="../output/processed_data/data_stage5.jsonl"

python refine_filter.py \
    --tag_jsonl $TAG_JSONL_FILE \
    --caption_jsonl $CAPTION_JSONL_FILE \
    --dover_vqa_jsonl $VQA_JSONL_FILE \
    --motion_score_jsonl $MOTION_SCORE_JSONL_FILE \
    --final_score_file $OUTPUT_SCORE_JSONL_FILE \
    --output_file $OUTPUT_JSONL_FILE \
    --top_k 25000
