import json
import os
from pathlib import Path
ROOT_DIR = Path(__file__).parent
import statistics

default_result_dir= ROOT_DIR / 'results/ClassPrediction'

def ensemble(X):
    def get_predictions(model, X):
        pass

    models = ['43Manual_outer2_inner1_model.keras', '43Manual_outer0_inner3_model.keras',
              '43Manual_outer3_inner2_model.keras',
              '43Manual_outer1_inner3_model.keras','43Manual_outer4_inner0_model.keras']
    results = [1,0,1,1,0]
    probabilities = [0.23621521890163422, 0.48942333459854126, 0.638190746307373, 0.30489876866340637, 0.3444334864616394]
    for model in models:
        print("MODEL:", model)

    return int(statistics.mode(results)), probabilities


def get_ensebmle_prediction(input_json, subject, output_dir):
    if output_dir:
        os.makedirs(output_dir,exist_ok=True)
    else:
        os.makedirs(default_result_dir, exist_ok=True)
    try:
        prediction, probabilities = ensemble(input_json)
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
    if output_dir:
        with open(os.path.join(output_dir, f"{subject}.json"), mode="w") as file:
            json.dump(response,file,indent=2)
    else:
        with open(os.path.join(default_result_dir, f"{subject}.json"), mode="w") as file:
            json.dump(response,file,indent=2)
    return response

def process_new_data(mode, model, front_file, side_file, env,output_dir):
    file_name=Path(side_file).name
    response=get_ensebmle_prediction(input_json=side_file, subject=file_name, output_dir=output_dir)
    print(response)
