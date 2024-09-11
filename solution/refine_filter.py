import argparse
import json
import os
import csv
import numpy as np
import random
random.seed(11)


def piecewise_linear(x, min_thresh=5, best_thresh=16, max_thresh=20):
    if x < min_thresh or x > max_thresh:
        return 0.
    elif min_thresh <= x <= best_thresh:
        return (x - min_thresh) / (best_thresh - min_thresh)
    elif best_thresh < x <= max_thresh:
        return 1.

def dover_score(aesthetic, technical, overall):
    # return aesthetic / 100. * 0.5 + technical / 100. * 0.5
    return overall / 100.

def filter_jsonl(caption_jsonl, tag_jsonl, dover_vqa_jsonl, motion_score_jsonl, output_score_file, keep_num, cond_max_keep=2, uncond_max_keep=2, uncond_ratio=0.5):
    captions = {}
    tags = {}
    dover_scores = {}
    scores = {}
    final_scores = []
    uncond_scores = dict()

    with open(caption_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            caption_data = json.loads(line.strip())
            captions[caption_data['videos'][0]] = [caption_data['text'], caption_data['confidence']]
    with open(tag_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tag_data = json.loads(line.strip())
            video_path = tag_data['videos'][0].replace('\/', '/')
            video_tag = ', '.join(tag_data['__dj__video_frame_tags__'][0])
            tags[video_path] = video_tag
    with open(dover_vqa_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            json_obj = json.loads(line.strip())
            video_path = json_obj['videos'][0]
            dover_scores[video_path] = [float(json_obj['aesthetic']), float(json_obj['technical']), float(json_obj['overall'])]
    with open(motion_score_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            json_obj = json.loads(line.strip())
            video_path = json_obj['video_path'].replace('\/', '/')
            scores[video_path] = [float(json_obj['motion_score'])]
            scores[video_path].extend(dover_scores[video_path])

    # get uncond items
    uncond_list = []
    uncond_num = int(uncond_ratio * keep_num)
    for k in captions.keys():
        v = scores[k]
        motion_score = v[0]
        dover_aesthetic_score = v[1] / 100.
        dover_technical_score = v[2] / 100.
        dover_overall_score = v[3] / 100.
        caption_confidence_score = captions[k][1]
        if dover_overall_score >= 0.3 and float(captions[k][1]) >= 0.75:
            uncond_list.append(k)
            uncond_scores[k] = [k, 0, motion_score, dover_aesthetic_score, dover_technical_score, dover_overall_score, caption_confidence_score]

    # keep only max_keep videos for videos splitted from a single video
    random.shuffle(uncond_list)
    vid_names = [os.path.basename(x).split('-')[0] for x in uncond_list]
    count_dict = {}
    keep_flag = []
    for vid_name in vid_names:
        if vid_name not in count_dict:
            count_dict[vid_name] = 0
        if count_dict[vid_name] < uncond_max_keep:
            keep_flag.append(True)
            count_dict[vid_name] += 1
        else:
            keep_flag.append(False)
    uncond_list = [item for item, flag in zip(uncond_list, keep_flag) if flag]
    uncond_list = uncond_list[:uncond_num]

    with open(output_score_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["videos", "final_score", "motion_score", "dover_score", "caption_confidence"])
        for k in uncond_list:
            item = uncond_scores[k]
            item = tuple(f"{p:.6f}" if isinstance(p, float) else p for p in item)
            writer.writerow([[item[0]], item[1], item[2], item[5], item[6]])

    # get cond items
    cond_list = list(captions.keys())
    cond_num = keep_num - uncond_num

    for k in cond_list:
        v = scores[k]
        motion_score = v[0]
        dover_aesthetic_score = v[1] / 100.
        dover_technical_score = v[2] / 100.
        dover_overall_score = v[3] / 100.
        caption_confidence_score = float(captions[k][1])
        if motion_score < 3 or motion_score > 30 or dover_overall_score < 0.3 or caption_confidence_score < 0.8:
            final_score = 0.
        else:
            final_score = piecewise_linear(motion_score) * 0.5 + dover_overall_score * 0.3 + caption_confidence_score * 0.2
        final_scores.append((k, final_score, motion_score, dover_aesthetic_score, dover_technical_score, dover_overall_score, caption_confidence_score))
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    # keep only max_keep videos for videos splitted from a single video
    vid_names = [os.path.basename(x[0]).split('-')[0] for x in final_scores]
    count_dict = {}
    keep_flag = []
    for vid_name in vid_names:
        if vid_name not in count_dict:
            count_dict[vid_name] = 0
        if count_dict[vid_name] < cond_max_keep:
            keep_flag.append(True)
            count_dict[vid_name] += 1
        else:
            keep_flag.append(False)
    final_scores = [item for item, flag in zip(final_scores, keep_flag) if flag][:cond_num]
    cond_list = [item[0] for item in final_scores]

    with open(output_score_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for item in final_scores:
            item = tuple(f"{p:.6f}" if isinstance(p, float) else p for p in item)
            writer.writerow([[item[0]], item[1], item[2], item[5], item[6]])

    return cond_list, uncond_list, captions, tags

def get_output_jsonl(cond_list, uncond_list, captions, tags, output_file, fps24=True):
    cond_list = sorted(cond_list)
    uncond_list = sorted(uncond_list)
    with open(output_file, 'w', encoding='utf-8') as f:
        for video_path in cond_list:
            caption = captions[video_path][0]
            if fps24:
                video_path = video_path.replace("videos_split", "videos_split_fps24")
            line = json.dumps({"videos": [video_path], "text": caption}, ensure_ascii=False) + '\n'
            f.write(line)
        for video_path in uncond_list:
            caption = "<__dj__video> <|__dj__eoc|>"
            if fps24:
                video_path = video_path.replace("videos_split", "videos_split_fps24")
            line = json.dumps({"videos": [video_path], "text": caption}, ensure_ascii=False) + '\n'
            f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter JSONL file by overall score threshold.')
    parser.add_argument("--tag_jsonl", help="Video tag jsonl path.") # not used
    parser.add_argument("--caption_jsonl", help="Video caption jsonl path.")
    parser.add_argument("--dover_vqa_jsonl", help="Dover VQA output jsonl path.")
    parser.add_argument("--motion_score_jsonl", help="Motion score jsonl path.")
    parser.add_argument('--final_score_file', help='Path to the output score JSONL file.')
    parser.add_argument('--output_file', help='Path to the output JSONL file.')
    parser.add_argument('--top_k', type=int, default=50000, help='Top k final score selected.')
    args = parser.parse_args()

    cond_list, uncond_list, captions, tags = filter_jsonl(args.caption_jsonl, args.tag_jsonl, args.dover_vqa_jsonl, args.motion_score_jsonl, args.final_score_file, keep_num=args.top_k)
    get_output_jsonl(cond_list, uncond_list, captions, tags, args.output_file)