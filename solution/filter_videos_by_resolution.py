import json
import cv2
import sys
from tqdm import tqdm

def filter_videos(input_jsonl, output_jsonl, min_width, min_height):
    with open(output_jsonl, 'w') as outfile:
        pass
    with open(input_jsonl, 'r') as infile, open(output_jsonl, 'a') as outfile:
        lines = infile.readlines()
        print('=== Input file first 10 lines:')
        print(''.join(lines[:10]))
        for i in tqdm(range(len(lines))):
            line = lines[i]
            data = json.loads(line.strip())
            video_path = data['videos'][0]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file {video_path}")
                continue
            
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if width >= min_width and height >= min_height:
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            cap.release()

input_jsonl = sys.argv[1]
output_jsonl = sys.argv[2]

min_width = 320
min_height = 320

filter_videos(input_jsonl, output_jsonl, min_width, min_height)
