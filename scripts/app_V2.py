import os
import sys
import re
import json
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import cv2
import joblib
from PIL import Image, ImageTk

from train_V3 import extract_features, apply_setting, normalize_setting_types


# =========================
# Path setup
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, "outputs")) and os.path.exists(os.path.join(BASE_DIR, "data")):
    PROJECT_DIR = BASE_DIR
else:
    PROJECT_DIR = os.path.dirname(BASE_DIR)


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


TRAIN_SCRIPT_PATH = first_existing_path(
    os.path.join(BASE_DIR, "train_V3.py"),
    os.path.join(PROJECT_DIR, "train_V3.py"),
    os.path.join(PROJECT_DIR, "scripts", "train_V3.py"),
)

EVAL_SCRIPT_PATH = first_existing_path(
    os.path.join(BASE_DIR, "evaluate.py"),
    os.path.join(PROJECT_DIR, "evaluate.py"),
    os.path.join(PROJECT_DIR, "scripts", "evaluate.py"),
)

MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "model.pkl")
CLASS_NAMES_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "class_names.pkl")
ACTIVE_CONFIG_PATH = os.path.join(PROJECT_DIR, "outputs", "models", "active_config_v3.json")
RESULT_CONFIG_PATH = os.path.join(PROJECT_DIR, "results", "best_config_v3.json")
EXAMPLES_DIR = os.path.join(PROJECT_DIR, "outputs", "examples")
METRICS_DIR = os.path.join(PROJECT_DIR, "outputs", "metrics")
APP_EVAL_SUMMARY_PATH = os.path.join(METRICS_DIR, "eval_summary_app.json")
WRONG_EXAMPLES_DIR = os.path.join(EXAMPLES_DIR, "wrong")

WINDOW_TITLE = "Campus Plant Recognition System"
WINDOW_SIZE = "1240x900"
WINDOW_MIN_SIZE = (1050, 760)
DISPLAY_IMAGE_SIZE = (360, 260)
IMAGE_BOX_PADDING = 20


# =========================
# Globals
# =========================
model = None
saved_class_names = []
current_image_path = None
current_process = None
log_queue = queue.Queue()
last_eval_summary = None


# =========================
# Basic helpers
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



def log_message(message):
    log_text.configure(state="normal")
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)
    log_text.configure(state="disabled")



def safe_read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def get_available_config_path():
    if os.path.exists(ACTIVE_CONFIG_PATH):
        return ACTIVE_CONFIG_PATH
    if os.path.exists(RESULT_CONFIG_PATH):
        return RESULT_CONFIG_PATH
    return None



def extract_setting_from_json(payload):
    if isinstance(payload, dict) and "best_setting" in payload:
        setting = payload["best_setting"]
    else:
        setting = payload
    return normalize_setting_types(setting)



def load_active_config():
    config_path = get_available_config_path()
    if config_path is None:
        raise FileNotFoundError("No config file found. Please run training first.")

    payload = safe_read_json(config_path)
    best_setting = extract_setting_from_json(payload)
    apply_setting(best_setting)
    config_path_label.config(text=f"Config: {os.path.basename(config_path)}")
    return best_setting, config_path



def load_model():
    global model, saved_class_names

    if not os.path.exists(MODEL_PATH):
        model = None
        saved_class_names = []
        model_status_label.config(text="Model status: not loaded")
        log_message("Model file not found.")
        return

    try:
        load_active_config()
        model = joblib.load(MODEL_PATH)
        if os.path.exists(CLASS_NAMES_PATH):
            saved_class_names = joblib.load(CLASS_NAMES_PATH)
        else:
            saved_class_names = []

        model_status_label.config(text="Model status: loaded")
        log_message("Model loaded successfully.")
    except Exception as e:
        model = None
        saved_class_names = []
        model_status_label.config(text="Model status: load failed")
        log_message(f"Failed to load model: {e}")



def open_config_file():
    config_path = get_available_config_path()
    if config_path is None:
        messagebox.showerror("Error", "No config file found.")
        log_message("Open config failed: no config file found.")
        return

    try:
        norm_path = os.path.normpath(config_path)
        if sys.platform.startswith("win"):
            subprocess.Popen(["notepad.exe", norm_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", norm_path])
        else:
            subprocess.Popen(["xdg-open", norm_path])
        log_message(f"Opened config file: {norm_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open config file.\n{e}")
        log_message(f"Open config failed: {e}")



def show_config_in_folder():
    config_path = get_available_config_path()
    if config_path is None:
        messagebox.showerror("Error", "No config file found.")
        log_message("Show in folder failed: no config file found.")
        return

    try:
        norm_path = os.path.normpath(config_path)
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", norm_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", norm_path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(norm_path)])
        log_message(f"Opened config folder: {norm_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot show config file in folder.\n{e}")
        log_message(f"Show in folder failed: {e}")



def set_task_buttons_state(is_running):
    state = tk.DISABLED if is_running else tk.NORMAL
    train_button.config(state=state)
    search_button.config(state=state)
    evaluate_button.config(state=state)
    reload_button.config(state=state)
    open_config_button.config(state=state)
    show_config_button.config(state=state)


# =========================
# Image display
# =========================
def cv_to_tk_image(image_bgr, target_size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image.thumbnail(target_size)
    return ImageTk.PhotoImage(pil_image)



def show_image_on_label(image_path, label_widget, target_size, empty_text):
    if not os.path.exists(image_path):
        label_widget.config(image="", text=empty_text, anchor="center", justify="center")
        label_widget.image = None
        return False

    image = cv2.imread(image_path)
    if image is None:
        label_widget.config(image="", text=empty_text, anchor="center", justify="center")
        label_widget.image = None
        return False

    tk_image = cv_to_tk_image(image, target_size)
    label_widget.config(image=tk_image, text="", anchor="center")
    label_widget.image = tk_image
    return True



def create_fixed_image_area(parent, placeholder_text):
    frame = tk.Frame(
        parent,
        width=DISPLAY_IMAGE_SIZE[0] + IMAGE_BOX_PADDING,
        height=DISPLAY_IMAGE_SIZE[1] + IMAGE_BOX_PADDING,
        relief="solid",
        bd=1,
        bg="white"
    )
    frame.pack_propagate(False)

    label = tk.Label(
        frame,
        text=placeholder_text,
        bg="white",
        anchor="center",
        justify="center"
    )
    label.pack(fill="both", expand=True)
    return frame, label



def show_input_image(filepath):
    ok = show_image_on_label(
        filepath,
        input_image_label,
        DISPLAY_IMAGE_SIZE,
        "Input image preview"
    )
    if ok:
        image_path_label.config(text=os.path.basename(filepath))
    else:
        image_path_label.config(text="Cannot display selected image")



def show_reference_image(pred_label):
    example_path = os.path.join(EXAMPLES_DIR, f"{pred_label}_example.jpg")
    ok = show_image_on_label(
        example_path,
        reference_image_label,
        DISPLAY_IMAGE_SIZE,
        "Reference image not found"
    )
    if ok:
        reference_title_label.config(text=f"Reference example: {pred_label}")
        log_message(f"Reference image loaded: {example_path}")
    else:
        reference_title_label.config(text="Reference example")
        log_message(f"Reference image not found for class: {pred_label}")


# =========================
# Prediction
# =========================
def predict_image(filepath):
    global model

    if model is None:
        messagebox.showwarning("Warning", "Model is not loaded.")
        log_message("Prediction cancelled: model is not loaded.")
        return

    try:
        load_active_config()
        image = cv2.imread(filepath)
        if image is None:
            result_label.config(text="Cannot read image")
            log_message(f"Cannot read image: {filepath}")
            return

        feature = extract_features(image).reshape(1, -1)

        if hasattr(model, "n_features_in_") and feature.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: image={feature.shape[1]}, model={model.n_features_in_}"
            )

        pred_label = model.predict(feature)[0]
        result_label.config(text=f"Predicted class: {pred_label}")
        log_message(f"Prediction completed: {pred_label}")
        show_reference_image(pred_label)
    except Exception as e:
        result_label.config(text="Prediction failed")
        log_message(f"Prediction failed: {e}")
        messagebox.showerror("Error", f"Prediction failed.\n{e}")



def choose_image():
    global current_image_path

    filepath = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )
    if filepath == "":
        return

    current_image_path = filepath
    show_input_image(filepath)
    predict_image(filepath)


# =========================
# Evaluation summary
# =========================
def update_eval_summary_widgets(summary):
    global last_eval_summary
    last_eval_summary = summary

    if not summary:
        eval_summary_label.config(
            text="Evaluation summary not available yet.\nClick 'Run Evaluation' to generate it."
        )
        view_eval_button.config(state=tk.DISABLED)
        open_wrong_button.config(state=tk.DISABLED)
        return

    accuracy = summary.get("accuracy")
    macro_f1 = summary.get("macro_f1")
    wrong_predictions = summary.get("wrong_predictions")
    total_samples = summary.get("total_samples")
    evaluated_at = summary.get("evaluated_at", "Unknown time")

    parts = [f"Last evaluation: {evaluated_at}"]
    if accuracy is not None:
        parts.append(f"Test accuracy: {accuracy:.3f}")
    if macro_f1 is not None:
        parts.append(f"Macro F1: {macro_f1:.3f}")
    if wrong_predictions is not None and total_samples is not None:
        parts.append(f"Wrong predictions: {wrong_predictions} / {total_samples}")

    eval_summary_label.config(text="\n".join(parts))
    view_eval_button.config(state=tk.NORMAL)
    open_wrong_button.config(state=tk.NORMAL if os.path.exists(WRONG_EXAMPLES_DIR) else tk.DISABLED)



def format_eval_details(summary):
    if not summary:
        return "No evaluation details available."

    details_parts = []
    if summary.get("confusion_matrix_text"):
        details_parts.append("Confusion matrix:\n" + summary["confusion_matrix_text"])
    if summary.get("classification_report_text"):
        details_parts.append("Classification report:\n" + summary["classification_report_text"])
    if not details_parts and summary.get("raw_output"):
        details_parts.append("Raw evaluation output:\n" + summary["raw_output"])

    return "\n\n".join(details_parts) if details_parts else "No evaluation details available."



def view_eval_details():
    if not last_eval_summary:
        messagebox.showinfo("Evaluation Details", "No evaluation summary is available yet.")
        return

    detail_window = tk.Toplevel(window)
    detail_window.title("Evaluation Details")
    detail_window.geometry("880x700")
    detail_window.minsize(760, 560)

    text_widget = tk.Text(detail_window, wrap="word")
    scrollbar = tk.Scrollbar(detail_window, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)

    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    text_widget.insert(tk.END, format_eval_details(last_eval_summary))
    text_widget.configure(state="disabled")



def open_wrong_examples():
    if not os.path.exists(WRONG_EXAMPLES_DIR):
        messagebox.showinfo("Wrong Examples", "Wrong examples folder does not exist yet.")
        return

    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", os.path.normpath(WRONG_EXAMPLES_DIR)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", WRONG_EXAMPLES_DIR])
        else:
            subprocess.Popen(["xdg-open", WRONG_EXAMPLES_DIR])
        log_message(f"Opened wrong examples folder: {WRONG_EXAMPLES_DIR}")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open wrong examples folder.\n{e}")
        log_message(f"Open wrong examples failed: {e}")



def save_eval_summary(summary):
    ensure_dir(METRICS_DIR)
    with open(APP_EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)



def load_saved_eval_summary():
    if not os.path.exists(APP_EVAL_SUMMARY_PATH):
        update_eval_summary_widgets(None)
        return

    try:
        with open(APP_EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        update_eval_summary_widgets(summary)
        log_message("Loaded saved evaluation summary.")
    except Exception as e:
        update_eval_summary_widgets(None)
        log_message(f"Failed to load saved evaluation summary: {e}")



def parse_evaluation_output(full_output):
    summary = {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_output": full_output.strip(),
    }

    acc_match = re.search(r"test accuracy is\s+([0-9.]+)", full_output, flags=re.IGNORECASE)
    if acc_match:
        summary["accuracy"] = float(acc_match.group(1))

    wrong_match = re.search(r"There are\s+(\d+)\s+wrong predictions", full_output, flags=re.IGNORECASE)
    if wrong_match:
        summary["wrong_predictions"] = int(wrong_match.group(1))

    accuracy_support_match = re.search(r"^\s*accuracy\s+([0-9.]+)\s+(\d+)\s*$", full_output, flags=re.MULTILINE)
    if accuracy_support_match:
        summary["report_accuracy"] = float(accuracy_support_match.group(1))
        summary["total_samples"] = int(accuracy_support_match.group(2))

    macro_match = re.search(r"^\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", full_output, flags=re.MULTILINE)
    if macro_match:
        summary["macro_precision"] = float(macro_match.group(1))
        summary["macro_recall"] = float(macro_match.group(2))
        summary["macro_f1"] = float(macro_match.group(3))
        if "total_samples" not in summary:
            summary["total_samples"] = int(macro_match.group(4))

    cm_match = re.search(r"confusion matrix:\s*\n(.*?)\nclassification report:", full_output, flags=re.IGNORECASE | re.DOTALL)
    if cm_match:
        summary["confusion_matrix_text"] = cm_match.group(1).strip()

    report_match = re.search(r"classification report:\s*\n(.*)$", full_output, flags=re.IGNORECASE | re.DOTALL)
    if report_match:
        summary["classification_report_text"] = report_match.group(1).strip()

    if "total_samples" in summary and "wrong_predictions" in summary and "correct_predictions" not in summary:
        summary["correct_predictions"] = summary["total_samples"] - summary["wrong_predictions"]

    return summary



def handle_evaluation_output(full_output, return_code):
    if return_code != 0:
        eval_summary_label.config(text="Evaluation failed. Check the log below for details.")
        log_message("Evaluation failed. Summary not updated.")
        return

    try:
        summary = parse_evaluation_output(full_output)
        save_eval_summary(summary)
        update_eval_summary_widgets(summary)
        log_message("Evaluation summary updated in the app.")
    except Exception as e:
        log_message(f"Failed to parse evaluation output: {e}")
        eval_summary_label.config(text="Evaluation finished, but summary parsing failed.")


# =========================
# Background task control
# =========================
def poll_log_queue():
    global current_process

    try:
        while True:
            item = log_queue.get_nowait()
            item_type = item.get("type")

            if item_type == "log":
                log_message(item["text"])

            elif item_type == "task_done":
                current_process = None
                set_task_buttons_state(False)

                task_name = item["task_name"]
                return_code = item["return_code"]
                completion_kind = item.get("completion_kind")
                full_output = item.get("full_output", "")

                log_message(f"{task_name} finished with return code {return_code}.")

                if completion_kind == "reload_model":
                    load_model()
                    log_message("Model reloaded after task completion.")
                elif completion_kind == "evaluation":
                    handle_evaluation_output(full_output, return_code)

    except queue.Empty:
        pass

    window.after(150, poll_log_queue)



def run_background_command(command, task_name, completion_kind=None):
    global current_process

    if current_process is not None:
        messagebox.showwarning("Warning", "Another task is already running.")
        log_message("Task request ignored: another task is already running.")
        return

    def worker():
        global current_process
        full_lines = []
        try:
            current_process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            if current_process.stdout is not None:
                for line in current_process.stdout:
                    clean_line = line.rstrip()
                    full_lines.append(clean_line)
                    log_queue.put({"type": "log", "text": clean_line})

            return_code = current_process.wait()
            log_queue.put({
                "type": "task_done",
                "task_name": task_name,
                "return_code": return_code,
                "completion_kind": completion_kind,
                "full_output": "\n".join(full_lines),
            })
        except Exception as e:
            log_queue.put({"type": "log", "text": f"{task_name} failed: {e}"})
            log_queue.put({
                "type": "task_done",
                "task_name": task_name,
                "return_code": -1,
                "completion_kind": completion_kind,
                "full_output": "\n".join(full_lines),
            })

    set_task_buttons_state(True)
    log_message(f"Starting: {task_name}")
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()



def train_with_current_config():
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        messagebox.showerror("Error", f"Cannot find train_V3.py at:\n{TRAIN_SCRIPT_PATH}")
        log_message("Training failed: train_V3.py not found.")
        return

    script_dir = os.path.dirname(os.path.abspath(TRAIN_SCRIPT_PATH))
    project_dir = os.path.dirname(script_dir)

    code = (
        "import os, sys, json\n"
        f"sys.path.insert(0, {script_dir!r})\n"
        "from train_V3 import augmented_cross_validation, train_final_model, normalize_setting_types\n"
        f"project_dir = {project_dir!r}\n"
        "config_candidates = [\n"
        "    os.path.join(project_dir, 'outputs', 'models', 'active_config_v3.json'),\n"
        "    os.path.join(project_dir, 'results', 'best_config_v3.json'),\n"
        "]\n"
        "config_path = next((p for p in config_candidates if os.path.exists(p)), None)\n"
        "if config_path is None:\n"
        "    raise FileNotFoundError('Cannot find active_config_v3.json or best_config_v3.json')\n"
        "with open(config_path, 'r', encoding='utf-8') as f:\n"
        "    payload = json.load(f)\n"
        "best_setting = payload['best_setting'] if isinstance(payload, dict) and 'best_setting' in payload else payload\n"
        "best_setting = normalize_setting_types(best_setting)\n"
        "augmented_cross_validation(best_setting)\n"
        "train_final_model(best_setting)\n"
    )

    command = [sys.executable, "-c", code]
    run_background_command(command, "Train with current config", completion_kind="reload_model")

def run_full_search():
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        messagebox.showerror("Error", f"Cannot find train_V3.py at:\n{TRAIN_SCRIPT_PATH}")
        log_message("Full search failed: train_V3.py not found.")
        return

    confirmed = messagebox.askyesno(
        "Confirm Full Search",
        "Run Full Search is not recommended for normal use.\n\n"
        "It may take about 2.5 hours to finish one full run.\n\n"
        "Do you still want to continue?"
    )
    if not confirmed:
        log_message("Full search cancelled by user.")
        return

    command = [
        sys.executable,
        "-c",
        (
            "from train_V3 import full_search_settings, save_search_results_csv, plot_search_results, save_best_config, augmented_cross_validation, train_final_model, SEARCH_CSV_PATH, SEARCH_PLOT_PATH, CONFIG_PATH; "
            "best_setting, best_score, results, search_space = full_search_settings(); "
            "save_search_results_csv(results, SEARCH_CSV_PATH); "
            "plot_search_results(results, SEARCH_PLOT_PATH); "
            "save_best_config(best_setting, best_score, results, search_space, CONFIG_PATH); "
            "augmented_cross_validation(best_setting); "
            "train_final_model(best_setting)"
        )
    ]
    run_background_command(command, "Run full search", completion_kind="reload_model")



def run_evaluation():
    if not os.path.exists(EVAL_SCRIPT_PATH):
        messagebox.showerror("Error", f"Cannot find evaluate.py at:\n{EVAL_SCRIPT_PATH}")
        log_message("Evaluation failed: evaluate.py not found.")
        return

    command = [sys.executable, EVAL_SCRIPT_PATH]
    run_background_command(command, "Run evaluation", completion_kind="evaluation")


# =========================
# UI construction
# =========================
window = tk.Tk()
window.title(WINDOW_TITLE)
window.geometry(WINDOW_SIZE)
window.minsize(WINDOW_MIN_SIZE[0], WINDOW_MIN_SIZE[1])


# Top area
header_frame = tk.Frame(window, padx=12, pady=10)
header_frame.pack(fill="x")

header_title = tk.Label(
    header_frame,
    text="Campus Plant Recognition System",
    font=("Arial", 18, "bold")
)
header_title.pack(anchor="w")

status_frame = tk.Frame(header_frame)
status_frame.pack(fill="x", pady=(6, 0))

model_status_label = tk.Label(status_frame, text="Model status: not loaded", anchor="w")
model_status_label.pack(side="left", padx=(0, 20))

config_path_label = tk.Label(status_frame, text="Config: not loaded", anchor="w")
config_path_label.pack(side="left")


# Main content area
content_frame = tk.Frame(window, padx=12, pady=6)
content_frame.pack(fill="both", expand=True)


# Image row
middle_frame = tk.Frame(content_frame)
middle_frame.pack(fill="x", pady=(0, 8))

left_frame = tk.LabelFrame(middle_frame, text="Input image", padx=10, pady=10)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

right_frame = tk.LabelFrame(middle_frame, text="Prediction result", padx=10, pady=10)
right_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

choose_button = tk.Button(left_frame, text="Choose Image", command=choose_image, width=18)
choose_button.pack(anchor="w")

image_path_label = tk.Label(left_frame, text="No image selected", anchor="w")
image_path_label.pack(anchor="w", pady=(8, 10))

input_image_box, input_image_label = create_fixed_image_area(left_frame, "Input image preview")
input_image_box.pack(fill="both", expand=True)

result_label = tk.Label(
    right_frame,
    text="Prediction will appear here",
    font=("Arial", 14, "bold"),
    anchor="w",
    justify="left"
)
result_label.pack(anchor="w", fill="x", pady=(0, 12))

reference_title_label = tk.Label(right_frame, text="Reference example", anchor="w")
reference_title_label.pack(anchor="w", pady=(0, 8))

reference_image_box, reference_image_label = create_fixed_image_area(
    right_frame,
    "Reference image will appear here"
)
reference_image_box.pack(fill="both", expand=True)


# Bottom row: control panel + evaluation summary (half / half)
bottom_row_frame = tk.Frame(content_frame)
bottom_row_frame.pack(fill="x", pady=(0, 8))

control_frame = tk.LabelFrame(bottom_row_frame, text="Model / Training / Evaluation", padx=10, pady=10)
control_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

control_left = tk.Frame(control_frame)
control_left.pack(side="left", fill="both", expand=True, padx=(0, 16))

control_right = tk.Frame(control_frame)
control_right.pack(side="left", fill="both", expand=True)

reload_button = tk.Button(control_left, text="Reload Model", width=24, command=load_model)
reload_button.pack(anchor="w", pady=4)

open_config_button = tk.Button(control_left, text="Open Config File", width=24, command=open_config_file)
open_config_button.pack(anchor="w", pady=4)

show_config_button = tk.Button(control_left, text="Show Config In Folder", width=24, command=show_config_in_folder)
show_config_button.pack(anchor="w", pady=4)

train_button = tk.Button(control_right, text="Train with Current Config", width=28, command=train_with_current_config)
train_button.pack(anchor="w", pady=4)

search_button = tk.Button(control_right, text="Run Full Search", width=28, command=run_full_search)
search_button.pack(anchor="w", pady=4)

evaluate_button = tk.Button(control_right, text="Run Evaluation", width=28, command=run_evaluation)
evaluate_button.pack(anchor="w", pady=4)


eval_frame = tk.LabelFrame(bottom_row_frame, text="Evaluation summary", padx=10, pady=10)
eval_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

eval_summary_label = tk.Label(
    eval_frame,
    text="Evaluation summary not available yet.\nClick 'Run Evaluation' to generate it.",
    anchor="nw",
    justify="left"
)
eval_summary_label.pack(anchor="w", fill="x", pady=(0, 10))

eval_action_frame = tk.Frame(eval_frame)
eval_action_frame.pack(anchor="w", fill="x")

view_eval_button = tk.Button(
    eval_action_frame,
    text="View Eval Details",
    width=18,
    command=view_eval_details,
    state=tk.DISABLED
)
view_eval_button.pack(side="left", padx=(0, 8))

open_wrong_button = tk.Button(
    eval_action_frame,
    text="Open Wrong Examples",
    width=18,
    command=open_wrong_examples,
    state=tk.DISABLED
)
open_wrong_button.pack(side="left")


# Log area
log_frame = tk.LabelFrame(content_frame, text="Log", padx=10, pady=10)
log_frame.pack(fill="both", expand=True, pady=(0, 6))

log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
log_text.pack(fill="both", expand=True)


# Initial state
load_model()
load_saved_eval_summary()
poll_log_queue()
window.mainloop()
