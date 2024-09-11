import json
import sys

thresh = 0.15

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(input_file_path, 'r') as infile:
    lines = infile.readlines()

filtered_data = []

for line in lines:
    data = json.loads(line.strip())
    if data['text_score'] < thresh:
        filtered_data.append({"videos": [data['video_path']], "text": "<__dj__video> <|__dj__eoc|>"})

with open(output_file_path, 'w') as outfile:
    for item in filtered_data:
        outfile.write(json.dumps(item) + '\n')