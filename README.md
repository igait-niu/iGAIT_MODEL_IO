
## iGAIT Command-Line Interface (CLI) Usage

This script provides a command-line interface to run the iGAIT pipeline for **training** or **prediction** using pose estimation data (OpenPose or MediaPipe).

### Basic Command Format

```bash
python main.py <mode> --model <openpose|mediapipe> --side <SIDE_PATH> [--front <FRONT_PATH>] [--env <DEV|PROD>]
```

## Arguments
1. Mode (required, positional)

```bash
<mode>
```
* Specifies the run mode.
* Choices:

  * `train` – for training the model
  * `predict` – for running inference

2. Model (required, named): Specifies which pose estimation model was used to generate the input data

```bash
--model openpose
```
or
```bash
--model mediapipe
```

3. Side View Input (required)

```bash
--side <path_to_side_view_json_or_folder>
```

* Path to the **side-view** JSON file or directory (depending on your pipeline design).
* Example:

```bash
--side /data/openpose-json/Su_229_S_6_0_X_X
```


4. Front View Input (optional)

```bash
--front <path_to_front_view_json_or_folder>
```

* Path to the **front-view** JSON file or directory (optional depending on your pipeline).
* Example:

```bash
--front /data/openpose-json/Su_229_F_6_0_X_X
```

5. Environment (optional)
* Controls runtime behavior:

  * `DEV`: saves intermediate files, prints debug logs
  * `PROD`: disables intermediate saving and debug outputs

If not provided, defaults to PROD and does not produce any intermediate files. To see the intermediate files and debug each stage in the pipeline switch it to DEV mode.

```bash
--env DEV
```

or

```bash
--env PROD
```

---

## Example Commands

### Prediction with OpenPose (Side View Only)

```bash
python main.py predict --model openpose --side /Users/you/iGait/data/openpose-json/Su_229_S_6_0_X_X --env PROD
```

### Prediction with OpenPose (Front + Side View)

```bash
python main.py predict --model openpose \
  --front /Users/you/iGait/data/openpose-json/Su_229_F_6_0_X_X \
  --side /Users/you/iGait/data/openpose-json/Su_229_S_6_0_X_X \
  --env DEV
```

### Training with MediaPipe

```bash
python main.py train --model mediapipe --side /data/mediapipe-json/subject_01 --env DEV
```

---

## Output Format

OUTPUT_DIR=results/ClassPrediction/

```
{
  "status": "success",
  "class": 1,
  "probabilities": [
    0.23621521890163422,
    0.48942333459854126,
    0.638190746307373,
    0.30489876866340637,
    0.3444334864616394
  ],
  "message": "Ensemble completed successfully"
}
```