import os
import csv
import json
import cv2
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =========================
# Global Settings
# =========================
IMG_SIZE = (128, 128)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 4423

# Search / output paths
RESULTS_DIR = "results"
OUTPUT_MODELS_DIR = os.path.join("outputs", "models")
CONFIG_PATH = os.path.join(RESULTS_DIR, "best_config_v3.json")
SEARCH_CSV_PATH = os.path.join(RESULTS_DIR, "feature_search_results_v3.csv")
SEARCH_PLOT_PATH = os.path.join(RESULTS_DIR, "feature_search_plot_v3.png")
AUG_CV_JSON_PATH = os.path.join(RESULTS_DIR, "augmented_cv_results_v3.json")

# Final defaults (used only if you manually call train_final_model without config)
COLOR_SPACE = "HSV"                 # "BGR" or "HSV"
COLOR_BINS = (16, 16, 16)
HOG_PIXELS_PER_CELL = (24, 24)
HOG_ORIENTATIONS = 9
USE_LBP = True
DEFAULT_BEST_PARAMS = {
    "classifier__C": 10,
    "classifier__gamma": 0.0001,
}


# =========================
# Utilities
# =========================
def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)



def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)



def normalize_setting_types(best_setting):
    """Convert JSON-loaded lists back to tuples where needed."""
    normalized = dict(best_setting)
    normalized["color_bins"] = tuple(normalized["color_bins"])
    normalized["pixels_per_cell"] = tuple(normalized["pixels_per_cell"])
    return normalized



def apply_setting(best_setting):
    """Apply one searched setting to global feature-extraction knobs."""
    global COLOR_SPACE, COLOR_BINS, HOG_PIXELS_PER_CELL, HOG_ORIENTATIONS, USE_LBP

    best_setting = normalize_setting_types(best_setting)
    COLOR_SPACE = best_setting["color_space"]
    COLOR_BINS = best_setting["color_bins"]
    HOG_PIXELS_PER_CELL = best_setting["pixels_per_cell"]
    HOG_ORIENTATIONS = best_setting["orientations"]
    USE_LBP = best_setting["use_lbp"]


# =========================
# Feature Extraction
# =========================
def extract_color(image):
    image = cv2.resize(image, IMG_SIZE)

    if COLOR_SPACE == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [converted],
            [0, 1, 2],
            None,
            COLOR_BINS,
            [0, 180, 0, 256, 0, 256]
        )
    elif COLOR_SPACE == "BGR":
        hist = cv2.calcHist(
            [image],
            [0, 1, 2],
            None,
            COLOR_BINS,
            [0, 256, 0, 256, 0, 256]
        )
    else:
        raise ValueError(f"unsupported COLOR_SPACE: {COLOR_SPACE}")

    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)



def extract_hog(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return features.astype(np.float32)



def extract_lbp(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist



def extract_features(image):
    feature_color = extract_color(image)
    feature_hog = extract_hog(image)

    if USE_LBP:
        feature_lbp = extract_lbp(image)
        feature = np.hstack([feature_color, feature_hog, feature_lbp]).astype(np.float32)
    else:
        feature = np.hstack([feature_color, feature_hog]).astype(np.float32)

    return feature


# =========================
# Data Augmentation
# =========================
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated



def adjust_brightness(image, factor):
    image_float = image.astype(np.float32) * factor
    image_float = np.clip(image_float, 0, 255)
    return image_float.astype(np.uint8)



def generate_augmented_images(image):
    return [
        image,
        rotate_image(image, 15),
        rotate_image(image, -15),
        cv2.flip(image, 1),
        adjust_brightness(image, 0.8),
        adjust_brightness(image, 1.2),
    ]


# =========================
# Dataset Reading
# =========================
def scan_dataset_paths(data_dir):
    """Read only file paths + labels first, so we can do fold splits cleanly."""
    class_names = []
    filepaths = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"cannot find directory: {data_dir}")
        return filepaths, np.array(labels), class_names

    for name in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, name)
        if os.path.isdir(full_path):
            class_names.append(name)

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for filename in sorted(os.listdir(class_dir)):
            filepath = os.path.join(class_dir, filename)
            if not os.path.isfile(filepath):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue
            filepaths.append(filepath)
            labels.append(class_name)

    return filepaths, np.array(labels), class_names



def build_feature_matrix_from_paths(filepaths, labels, augment=False, return_filepaths=False):
    features = []
    out_labels = []
    out_paths = []

    for filepath, label in zip(filepaths, labels):
        image = cv2.imread(filepath)
        if image is None:
            print(f"warning: cannot read {filepath}")
            continue

        image_list = generate_augmented_images(image) if augment else [image]

        for img in image_list:
            features.append(extract_features(img))
            out_labels.append(label)
            if return_filepaths:
                out_paths.append(filepath)

    features = np.array(features, dtype=np.float32)
    out_labels = np.array(out_labels)

    if return_filepaths:
        return features, out_labels, out_paths
    return features, out_labels



def read_dataset(data_dir, augment=False, return_filepaths=False):
    filepaths, labels, class_names = scan_dataset_paths(data_dir)

    if return_filepaths:
        features, labels_out, filepaths_out = build_feature_matrix_from_paths(
            filepaths, labels, augment=augment, return_filepaths=True
        )
        return features, labels_out, filepaths_out, class_names

    features, labels_out = build_feature_matrix_from_paths(
        filepaths, labels, augment=augment, return_filepaths=False
    )
    return features, labels_out, class_names


# =========================
# Model Building
# =========================
def build_search_model(param_grid=None):
    pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ])

    if param_grid is None:
        param_grid = {
            "classifier__C": [0.1, 1, 10, 50, 100],
            "classifier__gamma": ["scale", 0.01, 0.001, 0.0001, 0.00001]
        }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    return search



def build_final_model(best_params):
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=best_params["classifier__C"],
            gamma=best_params["classifier__gamma"]
        ))
    ])


# =========================
# Search Result Saving / Plotting
# =========================
def save_search_results_csv(results, save_path=SEARCH_CSV_PATH):
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "color_space",
            "color_bins",
            "pixels_per_cell",
            "orientations",
            "use_lbp",
            "feature_dim",
            "best_cv_accuracy",
            "best_C",
            "best_gamma"
        ])

        for row in results:
            writer.writerow([
                row["color_space"],
                row["color_bins"],
                row["pixels_per_cell"],
                row["orientations"],
                row["use_lbp"],
                row["feature_dim"],
                f'{row["best_cv_accuracy"]:.6f}',
                row["best_C"],
                row["best_gamma"]
            ])



def plot_search_results(results, save_path=SEARCH_PLOT_PATH):
    ensure_dir(os.path.dirname(save_path))

    x_labels = []
    seen = set()
    for row in results:
        x_label = f'{row["pixels_per_cell"][0]}x{row["pixels_per_cell"][1]} | ori={row["orientations"]}'
        if x_label not in seen:
            seen.add(x_label)
            x_labels.append(x_label)

    group_dict = {}
    for row in results:
        color_bin = row["color_bins"][0]
        group_name = f'{row["color_space"]} | bin={color_bin} | lbp={row["use_lbp"]}'
        x_label = f'{row["pixels_per_cell"][0]}x{row["pixels_per_cell"][1]} | ori={row["orientations"]}'

        if group_name not in group_dict:
            group_dict[group_name] = {}
        group_dict[group_name][x_label] = row["best_cv_accuracy"]

    plt.figure(figsize=(16, 8))
    for group_name, mapping in group_dict.items():
        y_values = [mapping.get(x, np.nan) for x in x_labels]
        linestyle = "--" if "lbp=True" in group_name else "-"
        plt.plot(x_labels, y_values, marker="o", linestyle=linestyle, label=group_name)

    plt.xlabel("HOG setting")
    plt.ylabel("Best 5-fold CV accuracy")
    plt.title("Feature search results (BGR vs HSV, color bins, HOG, LBP)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"plot saved to: {save_path}")



def save_best_config(best_setting, best_score, results, search_space, save_path=CONFIG_PATH):
    ensure_dir(os.path.dirname(save_path))

    payload = {
        "version": "train_V3",
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_cv_accuracy": float(best_score),
        "best_setting": best_setting,
        "search_space": search_space,
        "num_trials": len(results),
        "all_results": results,
        "csv_path": SEARCH_CSV_PATH,
        "plot_path": SEARCH_PLOT_PATH,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"best config saved to: {save_path}")



def load_best_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["best_setting"] = normalize_setting_types(payload["best_setting"])
    return payload


# =========================
# Feature Search
# =========================
def run_feature_search(color_space_options, color_bin_options,
                       pixel_options, orientation_options,
                       lbp_options, svm_param_grid):
    global COLOR_SPACE, COLOR_BINS, HOG_PIXELS_PER_CELL, HOG_ORIENTATIONS, USE_LBP

    train_dir = "data/train"
    best_score = -1
    best_setting = None
    results = []

    for color_space in color_space_options:
        for color_bin in color_bin_options:
            for pixels in pixel_options:
                for ori in orientation_options:
                    for use_lbp in lbp_options:
                        COLOR_SPACE = color_space
                        COLOR_BINS = (color_bin, color_bin, color_bin)
                        HOG_PIXELS_PER_CELL = pixels
                        HOG_ORIENTATIONS = ori
                        USE_LBP = use_lbp

                        print("\n==============================")
                        print("testing setting:")
                        print("COLOR_SPACE =", COLOR_SPACE)
                        print("COLOR_BINS =", COLOR_BINS)
                        print("pixels_per_cell =", HOG_PIXELS_PER_CELL)
                        print("orientations =", HOG_ORIENTATIONS)
                        print("use_lbp =", USE_LBP)

                        features_train, labels_train, _ = read_dataset(train_dir, augment=False)

                        print("train samples:", len(features_train))
                        print("feature shape:", features_train.shape)

                        search = build_search_model(param_grid=svm_param_grid)
                        search.fit(features_train, labels_train)

                        score = search.best_score_
                        best_params = search.best_params_

                        print(f"best cv accuracy: {score:.6f}")
                        print("best svm params:", best_params)

                        record = {
                            "color_space": COLOR_SPACE,
                            "color_bins": COLOR_BINS,
                            "pixels_per_cell": HOG_PIXELS_PER_CELL,
                            "orientations": HOG_ORIENTATIONS,
                            "use_lbp": USE_LBP,
                            "feature_dim": int(features_train.shape[1]),
                            "best_cv_accuracy": float(score),
                            "best_C": best_params["classifier__C"],
                            "best_gamma": best_params["classifier__gamma"]
                        }
                        results.append(record)

                        if score > best_score:
                            best_score = score
                            best_setting = {
                                "color_space": COLOR_SPACE,
                                "color_bins": COLOR_BINS,
                                "pixels_per_cell": HOG_PIXELS_PER_CELL,
                                "orientations": HOG_ORIENTATIONS,
                                "use_lbp": USE_LBP,
                                "svm_params": best_params
                            }

    print("\n==============================")
    print("final best setting:")
    print(best_setting)
    print(f"final best cv accuracy: {best_score:.6f}")

    return best_setting, best_score, results



def full_search_settings():
    color_space_options = ["BGR", "HSV"]
    color_bin_options = [8, 12, 16, 24]
    pixel_options = [(16, 16), (24, 24), (32, 32), (40, 40)]
    orientation_options = [9, 12, 15, 18]
    lbp_options = [False, True]

    svm_param_grid = {
        "classifier__C": [0.1, 1, 10, 50, 100],
        "classifier__gamma": ["scale", 0.01, 0.001, 0.0001, 0.00001]
    }

    search_space = {
        "color_space_options": color_space_options,
        "color_bin_options": color_bin_options,
        "pixel_options": pixel_options,
        "orientation_options": orientation_options,
        "lbp_options": lbp_options,
        "svm_param_grid": svm_param_grid,
    }

    best_setting, best_score, results = run_feature_search(
        color_space_options=color_space_options,
        color_bin_options=color_bin_options,
        pixel_options=pixel_options,
        orientation_options=orientation_options,
        lbp_options=lbp_options,
        svm_param_grid=svm_param_grid,
    )

    return best_setting, best_score, results, search_space


# =========================
# Augmented 5-fold CV + final training
# =========================
def augmented_cross_validation(best_setting, save_path=AUG_CV_JSON_PATH):
    apply_setting(best_setting)
    train_dir = "data/train"

    filepaths, labels, class_names = scan_dataset_paths(train_dir)
    if len(filepaths) == 0:
        print("training set is empty")
        return None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(filepaths, labels), start=1):
        train_paths = [filepaths[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_paths = [filepaths[i] for i in val_idx]
        val_labels = labels[val_idx]

        X_train, y_train = build_feature_matrix_from_paths(train_paths, train_labels, augment=True)
        X_val, y_val = build_feature_matrix_from_paths(val_paths, val_labels, augment=False)

        model = build_final_model(best_setting["svm_params"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        print(f"fold {fold_idx}: accuracy = {acc:.4f}")
        fold_accuracies.append(float(acc))
        fold_details.append({
            "fold": fold_idx,
            "train_original_samples": int(len(train_paths)),
            "train_augmented_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "accuracy": float(acc),
        })

    summary = {
        "setting": best_setting,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "num_classes": len(class_names),
        "details": fold_details,
    }

    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"augmented 5-fold CV mean accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"augmented CV results saved to: {save_path}")
    return summary



def train_final_model(best_setting=None):
    global COLOR_SPACE, COLOR_BINS, HOG_PIXELS_PER_CELL, HOG_ORIENTATIONS, USE_LBP

    set_random_seed()
    train_dir = "data/train"

    if best_setting is None:
        best_setting = {
            "color_space": COLOR_SPACE,
            "color_bins": COLOR_BINS,
            "pixels_per_cell": HOG_PIXELS_PER_CELL,
            "orientations": HOG_ORIENTATIONS,
            "use_lbp": USE_LBP,
            "svm_params": DEFAULT_BEST_PARAMS
        }

    apply_setting(best_setting)

    print("final training setting:")
    print("COLOR_SPACE =", COLOR_SPACE)
    print("COLOR_BINS =", COLOR_BINS)
    print("HOG_PIXELS_PER_CELL =", HOG_PIXELS_PER_CELL)
    print("HOG_ORIENTATIONS =", HOG_ORIENTATIONS)
    print("USE_LBP =", USE_LBP)
    print("best svm params =", best_setting["svm_params"])

    print("loading augmented training data...")
    features_train, labels_train, class_names = read_dataset(train_dir, augment=True)

    if len(features_train) == 0:
        print("training set is empty")
        return

    print("train samples:", len(features_train))
    print("feature shape:", features_train.shape)

    model = build_final_model(best_setting["svm_params"])
    model.fit(features_train, labels_train)

    ensure_dir(OUTPUT_MODELS_DIR)
    joblib.dump(model, os.path.join(OUTPUT_MODELS_DIR, "model.pkl"))
    joblib.dump(class_names, os.path.join(OUTPUT_MODELS_DIR, "class_names.pkl"))
    with open(os.path.join(OUTPUT_MODELS_DIR, "active_config_v3.json"), "w", encoding="utf-8") as f:
        json.dump(best_setting, f, indent=2)

    print("model saved to outputs/models/model.pkl")
    print("class names saved to outputs/models/class_names.pkl")
    print("active config saved to outputs/models/active_config_v3.json")


# =========================
# Main Entry
# =========================
def main():
    set_random_seed()

    if not os.path.exists(CONFIG_PATH):
        print("No best config file found. Start full exhaustive search...")
        best_setting, best_score, results, search_space = full_search_settings()
        save_search_results_csv(results, SEARCH_CSV_PATH)
        plot_search_results(results, SEARCH_PLOT_PATH)
        save_best_config(best_setting, best_score, results, search_space, CONFIG_PATH)
    else:
        print(f"Best config file found: {CONFIG_PATH}")
        payload = load_best_config(CONFIG_PATH)
        best_setting = payload["best_setting"]
        print("Loaded best setting:")
        print(best_setting)
        print(f"Stored best CV accuracy: {payload['best_cv_accuracy']:.6f}")

    print("\nStart augmented 5-fold training under the selected configuration...")
    augmented_cross_validation(best_setting)
    print("\nTrain final model on the full augmented training set...")
    train_final_model(best_setting)


if __name__ == "__main__":
    main()
