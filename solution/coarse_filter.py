import argparse
import json
import math

def get_jsonl_field(input_jsonl, key_field='video_path', value_field=None):
    rst = dict()
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            key = item[key_field]
            if isinstance(key, list):
                key = key[0]
            key = key.replace('\/', '/')
            rst[key] = item[value_field]
    return rst

def piecewise_linear(x, min_thresh=3, best_thresh=12, max_thresh=100):
    if x < min_thresh or x > max_thresh:
        return 0.
    elif x >= best_thresh:
        return 1.
    else:
        return 1/9*x - 1/3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter JSONL file by overall score threshold.')
    parser.add_argument('--input_jsonl', type=str, help='Path to the input JSONL file.')
    parser.add_argument('--text_score', type=str, help='Path to the text_score file.')
    parser.add_argument('--motion_score', type=str, help='Path to the motion_score file.')
    parser.add_argument('--vqa_score', type=str, help='Path to the vqa_score file.')
    parser.add_argument('--output_jsonl', type=str, help='Path to the output JSONL file.')
    parser.add_argument('--top_k', type=int, help='Choose top k items')

    args = parser.parse_args()

    text_scores = get_jsonl_field(args.text_score, value_field='text_score')
    motion_scores = get_jsonl_field(args.motion_score, value_field='motion_score')
    vqa_scores = get_jsonl_field(args.vqa_score, key_field='videos', value_field='overall')

    text_score_thresh = 0.1

    coarse_scores = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as infile, open(args.output_jsonl, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line.strip())
            video_path = item['videos'][0]
            text_score = float(text_scores[video_path])
            motion_score = float(motion_scores[video_path])
            vqa_score = float(vqa_scores[video_path]) / 100.

            if text_score >= text_score_thresh:
                coarse_score = 0
            else:
                coarse_score = math.sqrt(piecewise_linear(motion_score) * vqa_score)
            coarse_scores.append({'video_path': video_path, 'coarse_score': coarse_score})
        coarse_scores = sorted(coarse_scores, key=lambda x: x['coarse_score'], reverse=True)
        coarse_scores = coarse_scores[:args.top_k]
        coarse_scores = sorted(coarse_scores, key=lambda x: x['video_path'])
        for item in coarse_scores:
            filtered_item = {"videos": [item['video_path']], "coarse_score": f"{item['coarse_score']:.6f}", "text": "<__dj__video> <|__dj__eoc|>"}
            outfile.write(json.dumps(filtered_item) + '\n')
