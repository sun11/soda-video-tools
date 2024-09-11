import ast
import argparse
import gc
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm import tqdm

from utils.logger import logger
from utils.video_utils import get_video_path_list


@contextmanager
def VideoCapture(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        cap.release()
        del cap
        gc.collect()

def center_crop(image):
    height, width = image.shape[:2]

    side_length = min(height, width)
    center_x, center_y = width // 2, height // 2
    half_side = side_length // 2

    x1 = center_x - half_side
    y1 = center_y - half_side
    x2 = center_x + half_side
    y2 = center_y + half_side

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def _compute_resized_output_size(
    frame_size: Tuple[int, int],
    size: Optional[List[int]],
    max_size: Optional[int] = None,
) -> List[int]:
    h, w = frame_size
    short, long = (w, h) if w <= h else (h, w)

    if size is None:  # no change
        new_short, new_long = short, long
    elif len(size) == 1:  # specified size only for the smallest edge
        new_short = size[0]
        new_long = int(new_short * long / short)
    else:  # specified both h and w
        new_short, new_long = min(size), max(size)

    if max_size is not None and new_long > max_size:
        new_short = int(max_size * new_short / new_long)
        new_long = max_size

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return new_h, new_w

def compute_motion_score(video_path, args, use_center_crop=True):
    video_motion_scores = []
    normalized_video_motion_scores = []
    sampling_fps = 2

    try:
        with VideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            sampling_fps = min(sampling_fps, fps)
            sampling_step = round(fps / sampling_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # at least two frames for computing optical flow
            sampling_step = max(min(sampling_step, total_frames - 1),
                                1)

            prev_frame = None
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    # If the frame can't be read, it could be due to
                    # a corrupt frame or reaching the end of the video.
                    break

                height, width, _ = frame.shape
                new_h, new_w = _compute_resized_output_size(
                    (height, width), [args.frame_size], None)
                if (new_h, new_w) != (height, width):
                    frame = np.asarray(Image.fromarray(frame).resize((new_w, new_h)))

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if use_center_crop:
                    gray_frame = center_crop(gray_frame)

                if prev_frame is None:
                    prev_frame = gray_frame
                    continue

                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame,
                    gray_frame,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                frame_motion_score = np.mean(mag)
                normalized_frame_motion_score = frame_motion_score / np.hypot(*flow.shape[:2])
                video_motion_scores.append(frame_motion_score)
                normalized_video_motion_scores.append(normalized_frame_motion_score)
                prev_frame = gray_frame

                # quickly skip frames
                frame_count += sampling_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            video_meta_info = {
                "video_path": str(Path(video_path).resolve()),
                "motion_score": f"{float(np.mean(video_motion_scores or [-1])):.6f}",
                "normalized_motion_score": f"{float(np.mean(normalized_video_motion_scores or [-1])):.6f}",
            }
            return video_meta_info

    except Exception as e:
        print(f"Compute motion score for video {video_path} with error: {e}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the motion score of the videos.")
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument(
        "--video_metadata_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=100, help="The frequency to save the output results.")
    parser.add_argument("--n_jobs", type=int, default=1, help="The number of concurrent processes.")
    parser.add_argument(
        "--text_score_metadata_path", type=str, default=None, help="The path to the video text score metadata (csv/jsonl)."
    )
    parser.add_argument("--text_score_threshold", type=float, default=0.15, help="The text threshold.")
    parser.add_argument("--frame_size", type=int, default=512, help="The short side length of the video frame.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_path_list = get_video_path_list(
        video_folder=args.video_folder,
        video_metadata_path=args.video_metadata_path,
        video_path_column=args.video_path_column
    )
    print('=== 10 from video_path_list', video_path_list[:10])

    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")
    
    if os.path.exists(args.saved_path):
        print('=== args saved_path', args.saved_path)
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)
        saved_video_path_list = saved_metadata_df[args.video_path_column].tolist()
        saved_video_path_list = [os.path.join(args.video_folder, video_path) for video_path in saved_video_path_list]
        
        video_path_list = list(set(video_path_list).difference(set(saved_video_path_list)))
        # Sorting to guarantee the same result for each process.
        video_path_list = natsorted(video_path_list)
        logger.info(f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed.")
    
    if args.text_score_metadata_path is not None:
        if args.text_score_metadata_path.endswith(".csv"):
            text_score_df = pd.read_csv(args.text_score_metadata_path)
        elif args.text_score_metadata_path.endswith(".jsonl"):
            text_score_df = pd.read_json(args.text_score_metadata_path, lines=True)

        filtered_text_score_df = text_score_df[text_score_df["text_score"] >= args.text_score_threshold]
        filtered_video_path_list = filtered_text_score_df[args.video_path_column].tolist()
        filtered_video_path_list = [os.path.join(args.video_folder, video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        # Sorting to guarantee the same result for each process.
        video_path_list = natsorted(video_path_list)
        logger.info(f"Load {args.text_score_metadata_path} and filter {len(filtered_video_path_list)} videos.")

    for i in tqdm(range(0, len(video_path_list), args.saved_freq)):
        result_list = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(compute_motion_score)(video_path, args) for video_path in tqdm(video_path_list[i: i + args.saved_freq])
        )
        result_list = [result for result in result_list if result is not None]
        if len(result_list) == 0:
            continue

        result_df = pd.DataFrame(result_list)
        if args.saved_path.endswith(".csv"):
            header = False if os.path.exists(args.saved_path) else True
            result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
        elif args.saved_path.endswith(".jsonl"):
            result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
        logger.info(f"Save result to {args.saved_path}.")


if __name__ == "__main__":
    main()