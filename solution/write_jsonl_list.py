import os
import json
import sys

def write_filelist_to_jsonl(folder_path, output_file):
    mp4_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    
    with open(output_file, 'w') as jsonl_file:
        for vidfile in mp4_files:
            entry = {"videos": [os.path.abspath(os.path.join(folder_path, vidfile))], "text": "<__dj__video> <|__dj__eoc|>"}
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python write_jsonl_list.py input_dir output.jsonl"
    write_filelist_to_jsonl(sys.argv[1], sys.argv[2])