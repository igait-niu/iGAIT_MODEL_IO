import json
import os
from pathlib import Path
ROOT_DIR = Path(__file__).parent
import statistics

default_result_dir = ROOT_DIR / 'results/ClassPrediction'


def load_gait_analysis(file_path):
    """Load a gait analysis JSON file and return (gait_cycles, landmark_data).

    Expected JSON format:
    {
        "gait_cycles": [
            { "start": <int>, "end": <int>, "side": "L"|"R" },
            ...
        ],
        "landmark_data": [
            {
                "frame_number": <int>,
                "timestamp": <float>,
                "pose_landmarks": [[x, y, z], ...] | null,       // 33 landmarks
                "left_hand_landmarks": [[x, y, z], ...] | null,  // 21 landmarks (Shaivil will remove later)
                "right_hand_landmarks": [[x, y, z], ...] | null, // 21 landmarks (Shaivil will remove later)
                "face_landmarks": [[x, y, z], ...] | null        // 478 landmarks (Shaivil will remove later)
            },
            ...
        ]
    }
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    gait_cycles = data["gait_cycles"]
    landmark_data = data["landmark_data"]

    return gait_cycles, landmark_data


def print_gait_summary(label, gait_cycles, landmark_data):
    """Print a human-readable summary of loaded gait analysis data."""
    print(f"\n{'='*60}")
    print(f"  {label} Gait Analysis Summary")
    print(f"{'='*60}")

    print(f"  Total frames: {len(landmark_data)}")
    frames_with_pose = sum(1 for f in landmark_data if f.get('pose_landmarks') is not None)
    print(f"  Frames with pose data: {frames_with_pose}")
    print(f"  Frames without pose data: {len(landmark_data) - frames_with_pose}")

    print(f"\n  Gait cycles ({len(gait_cycles)} total):")
    for i, cycle in enumerate(gait_cycles):
        print(f"    Cycle {i+1}: frames {cycle['start']}-{cycle['end']} (side={cycle['side']}, length={cycle['end'] - cycle['start']} frames)")

    # Show a sample frame with data
    for frame in landmark_data:
        if frame.get('pose_landmarks') is not None:
            print(f"\n  Sample frame (frame #{frame['frame_number']}):")
            print(f"    pose_landmarks: {len(frame['pose_landmarks'])} landmarks (33 expected)")
            if frame.get('left_hand_landmarks'):
                print(f"    left_hand_landmarks: {len(frame['left_hand_landmarks'])} landmarks")
            else:
                print(f"    left_hand_landmarks: null")
            if frame.get('right_hand_landmarks'):
                print(f"    right_hand_landmarks: {len(frame['right_hand_landmarks'])} landmarks")
            else:
                print(f"    right_hand_landmarks: null")
            if frame.get('face_landmarks'):
                print(f"    face_landmarks: {len(frame['face_landmarks'])} landmarks")
            else:
                print(f"    face_landmarks: null")
            break

    print(f"{'='*60}\n")


def ensemble(X):
    """Placeholder ensemble prediction - to be implemented by ML team."""
    def get_predictions(model, X):
        pass

    models = ['43Manual_outer2_inner1_model.keras', '43Manual_outer0_inner3_model.keras',
              '43Manual_outer3_inner2_model.keras',
              '43Manual_outer1_inner3_model.keras', '43Manual_outer4_inner0_model.keras']
    results = [1, 0, 1, 1, 0]
    probabilities = [0.23621521890163422, 0.48942333459854126, 0.638190746307373, 0.30489876866340637, 0.3444334864616394]
    for model in models:
        print("MODEL:", model)

    return int(statistics.mode(results)), probabilities


def get_ensemble_prediction(front_gait_cycles, front_landmark_data,
                            side_gait_cycles, side_landmark_data,
                            subject, output_dir):
    """Run ensemble prediction and write result to output_dir.

    Args:
        front_gait_cycles: List of gait cycle dicts from front view
        front_landmark_data: List of frame dicts from front view
        side_gait_cycles: List of gait cycle dicts from side view
        side_landmark_data: List of frame dicts from side view
        subject: Subject identifier for output filename
        output_dir: Directory to write prediction result JSON
    """
    result_dir = output_dir if output_dir else default_result_dir
    os.makedirs(result_dir, exist_ok=True)

    try:
        # TODO (ML team): Replace with actual feature extraction + ensemble prediction
        #   using front/side gait_cycles and landmark_data.
        prediction, probabilities = ensemble(None)
        response = {
            "status": "success",
            "class": prediction,
            "probabilities": probabilities,
            "message": "Ensemble completed successfully"
        }
    except Exception as e:
        response = {
            "status": "error",
            "stage": "ensemble prediction",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

    output_path = os.path.join(result_dir, f"{subject}.json")
    with open(output_path, mode="w") as file:
        json.dump(response, file, indent=2)

    return response


def process_new_data(mode, model, front_file, side_file, env, output_dir):
    # --- Load and display side gait analysis ---
    print(f"\nLoading side gait analysis from: {side_file}")
    side_gait_cycles, side_landmark_data = load_gait_analysis(side_file)
    print_gait_summary("Side", side_gait_cycles, side_landmark_data)

    # --- Load and display front gait analysis (if provided) ---
    front_gait_cycles = []
    front_landmark_data = []
    if front_file:
        print(f"Loading front gait analysis from: {front_file}")
        front_gait_cycles, front_landmark_data = load_gait_analysis(front_file)
        print_gait_summary("Front", front_gait_cycles, front_landmark_data)
    else:
        print("No front gait analysis file provided, skipping.")

    # --- Run prediction ---
    subject = Path(side_file).stem
    response = get_ensemble_prediction(
        front_gait_cycles=front_gait_cycles,
        front_landmark_data=front_landmark_data,
        side_gait_cycles=side_gait_cycles,
        side_landmark_data=side_landmark_data,
        subject=subject,
        output_dir=output_dir,
    )
    print(response)
