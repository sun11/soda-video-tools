# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import glob
import sys
import json
import torch
import numpy as np
from copy import deepcopy
from transformers import StoppingCriteriaList
from tasks.utils import load_model_and_processor
from dataset.utils import *
from tools.conversation import Chat, conv_templates, StoppingCriteriaSub
from transformers import TextStreamer
from tools.color import Color


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Load Model
    print(f"### Start loading model...")
    model, processor = load_model_and_processor(args.model_name_or_path, max_n_frames=args.max_n_frames)
    print(f"### Finish loading model.")
    if '7b' in args.model_name_or_path.lower():
        conv_type = 'tarsier-7b'
    elif '13b' in args.model_name_or_path.lower():
        conv_type = 'tarsier-13b'
    elif '34b' in args.model_name_or_path.lower():
        conv_type = 'tarsier-34b'
    else:
        raise ValueError(f"Unknow model: {args.model_name_or_path}")

    chat = Chat(model, processor, device=device, debug = args.debug)
    conv = deepcopy(conv_templates[conv_type])


    filelist = []
    json_data = []

    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip('\n'))
            json_data.append(json_obj)
            filelist.append(json_obj['videos'][0])
    print('=== 10 files in the front of filelist:', len(filelist))
    print(filelist[:10])

    with open(args.output_path, 'w') as f:
        pass

    images = []
    for i, img_path in enumerate(filelist):
        print('=== Processing', img_path, f'{(i+1)/len(filelist)*100:.2f} %')
        inp = 'Describe the video in detail beginning with "The video shows"'
        conv = chat.ask(inp, conv)
        # print(Color.green(f"{conv.roles[1]}: "), end="")

        stop_words_ids = [torch.tensor([processor.tokenizer.eos_token_id]).to(device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs, conv, images = chat.prepare_model_inputs(conv, img_path, images, args.max_n_frames)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                streamer=None,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        scores = [torch.amax(torch.nn.functional.softmax(input=s, dim=-1)).cpu() for s in outputs.scores]
        caption_confidence = np.mean(scores)

        outputs = processor.tokenizer.decode(outputs.sequences[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        prefix = 'The video shows '
        if outputs.startswith(prefix):
            outputs = outputs[len(prefix):]
        print(Color.green(f'{caption_confidence:.6f}, {outputs}'))
        # conv.messages[-1][1] = outputs
        with open(args.output_path, 'a') as f:
            json_obj = json_data[i]
            json_obj['text'] = '<__dj__video> ' + outputs + ' <|__dj__eoc|>'
            json_obj['confidence'] = f'{caption_confidence:.10f}'
            json_str = json.dumps(json_obj, ensure_ascii=False)
            f.write(json_str + '\n')

        if args.debug:
            print(f"Conversation state: {conv}")
        conv = deepcopy(conv_templates[conv_type])
        images = []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--model_name_or_path', type=str, default="/home/starsky/models/hf/Tarsier-7b")
    parser.add_argument("--max_n_frames", type=int, default=8, help="Max number of frames to apply average sampling from the given video.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="max number of generated tokens")
    parser.add_argument("--top_p", type=float, default=1, help="Top_p sampling")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
