import os
import argparse
from Core import FrameExtractor

def main(videos_path:str, frames_path:str, args):
    # Frame Extraction
    FrameExtractor.run(
        videos_path=videos_path,
        save_path=frames_path,
        frame_size=args.frame_size,
        workers=args.workers,
        original_size=args.no_original_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--dataset-name", type=str, default="UCF101", choices={"UCF101", "HMDB51", "ActivityNet"})
    parser.add_argument("--split-id", type=int, default=1)
    # For Frame Extractor
    parser.add_argument("--frame-size", type=int, default=240)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--no-original-size", action="store_false")
    args = parser.parse_args()

    # Path organize
    videos_path = os.path.join(args.data_path, f"{args.dataset_name}/videos/")
    frames_path = os.path.join(args.data_path, f"{args.dataset_name}/frames/")

    main(videos_path, frames_path, args)
