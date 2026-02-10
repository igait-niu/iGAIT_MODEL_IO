
import argparse

from predict import process_new_data


def parse_args():
    parser = argparse.ArgumentParser(prog="iGAIT", description="Autism Detection on side walk",
                                     epilog="Txt at the bottom of the help")
    parser.add_argument("mode", choices=["train", "predict"], help="Run Mode")
    parser.add_argument("--model", choices=["openpose", "mediapipe"], required=True, help="Pose estimation model")
    parser.add_argument("--front",type=str, help="Front-view JSON file path")
    parser.add_argument("--side", type=str, required=True, help="Side-view JSON file path")
    parser.add_argument("--env", choices=["DEV", "PROD"], default=None)
    args = parser.parse_args()
    print(f"args:{args}")

    return args


def main():
    args = parse_args()
    print("Running in mode:",args.env)
    process_new_data(mode=args.mode, model=args.model, front_file=args.front, side_file=args.side,
                     env=args.env)
    # ensure_requirements(req_file="requirements.txt")


if __name__ == "__main__":
    main()
