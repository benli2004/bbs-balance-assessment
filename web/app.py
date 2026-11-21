from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path
import tempfile
import shutil
import yaml
import glob

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_preprocessing.frame_extractor import extract_frames
from models.bbsaqa_3dcnn import BBSAQA3DCNN

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configuration
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Task names for the different tasks
TASK_NAMES = [
    "Task1 - 坐到站立",
    "Task2 - 站立2分鐘",
    "Task3 - 雙臂交叉坐姿",
    "Task4 - 站立到坐下",
    "Task5 - 撿東西",
    "Task6 - 單腳站立",
    "Task7 - 走台階"
]

# Global model instance
model = None
device = None
data_cfg = None
model_cfg = None
model_loaded = False


def load_config():
    global data_cfg, model_cfg
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    with open(DATA_CONFIG_PATH, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]


def find_checkpoint():
    """Find checkpoint file in checkpoints directory."""
    if not CHECKPOINTS_DIR.exists():
        return None
    
    # Look for .pth files
    pth_files = list(CHECKPOINTS_DIR.glob("*.pth"))
    if pth_files:
        return str(pth_files[0])  # Return first found checkpoint
    return None


def load_model(ckpt_path: str):
    global model, device, model_loaded
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BBSAQA3DCNN(
        num_actions=model_cfg["num_actions"],
        num_scores=model_cfg["num_scores"],
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    model_loaded = True
    return model


def predict_video(video_path: str) -> dict:
    """Predict score for a single video."""
    if model is None or not model_loaded:
        raise RuntimeError("Model not loaded. No checkpoint found.")
    
    target_frames = data_cfg["target_frames"]
    resize_h, resize_w = data_cfg["resize"]
    resize_h, resize_w = int(resize_h), int(resize_w)
    
    video_tensor = extract_frames(
        video_path,
        target_frames=target_frames,
        resize=(resize_h, resize_w),
        decode="cpu",
    )  # (C, T, H, W)
    
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)
    
    with torch.no_grad():
        outputs = model(video_tensor)
        
        action_logits = outputs["action_logits"].squeeze(0)
        pred_action_id = torch.argmax(action_logits).item() + 1
        
        action_scores = outputs["action_scores"].squeeze(0)
        pred_action_score_logits = action_scores[pred_action_id - 1]
        pred_score = torch.argmax(pred_action_score_logits).item()
        
        # Get confidence scores
        action_probs = torch.softmax(action_logits, dim=0)
        score_probs = torch.softmax(pred_action_score_logits, dim=0)
        
        pred_action_confidence = action_probs[pred_action_id - 1].item()
        pred_score_confidence = score_probs[pred_score].item()
    
    return {
        "action_id": pred_action_id,
        "action_confidence": pred_action_confidence,
        "score": pred_score,
        "score_confidence": pred_score_confidence,
    }


@app.route('/')
def index():
    return render_template('index.html', tasks=TASK_NAMES, model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle video uploads and return predictions for all 7 tasks."""
    
    # Check if model is loaded
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please train the model first or add a checkpoint to the checkpoints folder."}), 400
    
    results = []
    
    # Create temp directory for uploaded videos
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Check if files were uploaded
        if 'videos' not in request.files:
            return jsonify({"error": "No video files uploaded"}), 400
        
        files = request.files.getlist('videos')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No video files selected"}), 400
        
        # Process each uploaded video
        for idx, file in enumerate(files):
            if file.filename == '':
                continue
            
            # Save uploaded file
            video_path = os.path.join(temp_dir, f"video_{idx}.mp4")
            file.save(video_path)
            
            try:
                # Run prediction
                prediction = predict_video(video_path)
                results.append({
                    "task_index": idx,
                    "task_name": TASK_NAMES[idx] if idx < len(TASK_NAMES) else f"Task {idx + 1}",
                    "video_name": file.filename,
                    "score": prediction["score"],
                    "score_label": get_score_label(prediction["score"]),
                    "confidence": prediction["score_confidence"],
                })
            except Exception as e:
                results.append({
                    "task_index": idx,
                    "task_name": TASK_NAMES[idx] if idx < len(TASK_NAMES) else f"Task {idx + 1}",
                    "video_name": file.filename,
                    "error": str(e),
                })
        
        # Sort by score (descending) for ranking
        valid_results = [r for r in results if "error" not in r]
        valid_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank to each result
        for rank, result in enumerate(valid_results, 1):
            result["rank"] = rank
        
        return jsonify({
            "results": results,
            "ranked_results": valid_results,
        })
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Handle a single video upload and return predictions for all 7 tasks."""
    
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 400
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        video_path = os.path.join(temp_dir, "video.mp4")
        file.save(video_path)
        
        prediction = predict_video(video_path)
        score = prediction["score"]
        
        # Generate results for all 7 tasks with same score (for demo)
        # In production, you might want to process video segments or use multiple runs
        results = []
        for i in range(7):
            results.append({
                "task_index": i,
                "score": score,
                "score_label": get_score_label(score),
                "confidence": prediction["score_confidence"],
            })
        
        return jsonify({
            "results": results,
            "total_score": score * 7,
            "avg_score": score,
            "risk_level": "high" if score <= 2 else "medium" if score <= 3 else "low",
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Handle a single video upload and return prediction for one task."""
    
    # Check if model is loaded
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 400
    
    # Check if file was uploaded
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    file = request.files['video']
    task_index = request.form.get('task_index', type=int, default=0)
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        video_path = os.path.join(temp_dir, "video.mp4")
        file.save(video_path)
        
        # Run prediction
        prediction = predict_video(video_path)
        
        return jsonify({
            "task_index": task_index,
            "task_name": TASK_NAMES[task_index] if task_index < len(TASK_NAMES) else f"Task {task_index + 1}",
            "video_name": file.filename,
            "score": prediction["score"],
            "confidence": prediction["score_confidence"],
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_score_label(score: int) -> str:
    """Convert numeric score to label."""
    labels = {
        0: "Very Poor",
        1: "Poor", 
        2: "Average",
        3: "Good",
        4: "Excellent"
    }
    return labels.get(score, f"Score {score}")


if __name__ == '__main__':
    # Load configurations
    load_config()
    
    # Try to find and load checkpoint automatically
    checkpoint_path = find_checkpoint()
    if checkpoint_path:
        try:
            load_model(checkpoint_path)
            print("[OK] Model loaded from: " + checkpoint_path)
        except Exception as e:
            print("[ERROR] Failed to load model: " + str(e))
            print("  The model will not be available for predictions.")
            print("  You can still use the web interface but predictions will fail.")
    else:
        print("[INFO] No checkpoint found in: " + str(CHECKPOINTS_DIR))
        print("  Please train the model first or add a .pth file to the checkpoints folder.")
        print("  Expected path: checkpoints/bbsaqa3dcnn_last.pth")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
