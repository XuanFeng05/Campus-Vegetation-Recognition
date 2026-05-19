import os
import csv
import cv2
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Global Settings
IMG_SIZE = (128, 128)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 4423

# Fixed color setting for V2
COLOR_BINS = (12,12,12)

# Default final feature setting
HOG_PIXELS_PER_CELL = (24,24)
HOG_ORIENTATIONS = 9
USE_LBP = True

# Default final SVM setting
DEFAULT_BEST_PARAMS = {
    "classifier__C": 10,
    "classifier__gamma": 0.0001
}



# Utilities
def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Feature Extraction
def extract_color(image):
    image = cv2.resize(image, IMG_SIZE)
    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        COLOR_BINS,
        [0, 256, 0, 256, 0, 256]
    )
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

# Data Augmentation
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

# Dataset Reading
def read_dataset(data_dir, augment=False, return_filepaths=False):
    features = []
    labels = []
    class_names = []
    filepaths=[]

    # if not os.path.exists(data_dir):
    #     print(f"cannot find directory: {data_dir}")
    #     return np.array(features, dtype=np.float32), np.array(labels), class_names
    if not os.path.exists(data_dir):
        print(f"cannot find directory: {data_dir}")
        if return_filepaths:
            return np.array(features, dtype=np.float32), np.array(labels), filepaths, class_names
        return np.array(features, dtype=np.float32), np.array(labels), class_names
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

            image = cv2.imread(filepath)
            if image is None:
                print(f"warning: cannot read {filepath}")
                continue

            image_list = generate_augmented_images(image) if augment else [image]

            for img in image_list:
                features.append(extract_features(img))
                labels.append(class_name)
                if return_filepaths:
                    filepaths.append(filepath)
    if return_filepaths:
        return np.array(features, dtype=np.float32), np.array(labels), filepaths, class_names
    return np.array(features, dtype=np.float32), np.array(labels), class_names
# Model Building
def build_search_model(param_grid=None):
    pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ])

    if param_grid is None:
        param_grid = {
            "classifier__C": [0.1, 1, 10, 50],
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

# Search Result Saving / Plotting
def save_search_results_csv(results, save_path="results/feature_search_results.csv"):
    ensure_dir(os.path.dirname(save_path))

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
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
                row["color_bins"],
                row["pixels_per_cell"],
                row["orientations"],
                row["use_lbp"],
                row["feature_dim"],
                f'{row["best_cv_accuracy"]:.6f}',
                row["best_C"],
                row["best_gamma"]
            ])


def plot_search_results(results, save_path="results/feature_search_plot.png"):
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
        color_bin = row["color_bins"][0]   # (12,12,12) -> 12
        group_name = f'bin={color_bin}, lbp={row["use_lbp"]}'
        x_label = f'{row["pixels_per_cell"][0]}x{row["pixels_per_cell"][1]} | ori={row["orientations"]}'

        if group_name not in group_dict:
            group_dict[group_name] = {}

        group_dict[group_name][x_label] = row["best_cv_accuracy"]

    plt.figure(figsize=(12, 6))

    for group_name, mapping in group_dict.items():
        y_values = [mapping.get(x, np.nan) for x in x_labels]

        linestyle = "--" if "lbp=True" in group_name else "-"
        plt.plot(x_labels, y_values, marker="o", linestyle=linestyle, label=group_name)

    plt.xlabel("HOG setting")
    plt.ylabel("Best 5-fold CV accuracy")
    plt.title("Feature search results")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"plot saved to: {save_path}")

# Feature Search
def run_feature_search(color_bin_options, pixel_options, orientation_options, lbp_options, svm_param_grid):
    global COLOR_BINS, HOG_PIXELS_PER_CELL, HOG_ORIENTATIONS, USE_LBP

    train_dir = "data/train"
    best_score = -1
    best_setting = None
    results = []

    for color_bin in color_bin_options:
        for pixels in pixel_options:
            for ori in orientation_options:
                for use_lbp in lbp_options:
                    COLOR_BINS = (color_bin, color_bin, color_bin)
                    HOG_PIXELS_PER_CELL = pixels
                    HOG_ORIENTATIONS = ori
                    USE_LBP = use_lbp

                    print("\n==============================")
                    print("testing setting:")
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

    return best_setting, results

def quick_search_settings():
    color_bin_options = [12, 16, 24]
    pixel_options = [(24, 24), (32, 32)]
    orientation_options = [9, 15]
    lbp_options = [False, True]

    svm_param_grid = {
        "classifier__C": [1, 10],
        "classifier__gamma": ["scale", 0.01]
    }

    return run_feature_search(
        color_bin_options=color_bin_options,
        pixel_options=pixel_options,
        orientation_options=orientation_options,
        lbp_options=lbp_options,
        svm_param_grid=svm_param_grid
    )

def full_search_settings():
    color_bin_options = [12, 16, 24]
    pixel_options = [(24, 24), (32, 32)]
    orientation_options = [9, 12, 15, 18]
    lbp_options = [False, True]

    svm_param_grid = {
        "classifier__C": [0.1, 1, 10, 50],
        "classifier__gamma": ["scale", 0.01, 0.001, 0.0001]
    }

    return run_feature_search(
        color_bin_options=color_bin_options,
        pixel_options=pixel_options,
        orientation_options=orientation_options,
        lbp_options=lbp_options,
        svm_param_grid=svm_param_grid
    )

# Final Training
def train_final_model(best_setting=None):
    global COLOR_BINS, HOG_PIXELS_PER_CELL, HOG_ORIENTATIONS, USE_LBP

    set_random_seed()
    train_dir = "data/train"

    if best_setting is None:
        best_setting = {
            "color_bins": COLOR_BINS,
            "pixels_per_cell": HOG_PIXELS_PER_CELL,
            "orientations": HOG_ORIENTATIONS,
            "use_lbp": USE_LBP,
            "svm_params": DEFAULT_BEST_PARAMS
        }
    COLOR_BINS = best_setting["color_bins"]
    HOG_PIXELS_PER_CELL = best_setting["pixels_per_cell"]
    HOG_ORIENTATIONS = best_setting["orientations"]
    USE_LBP = best_setting["use_lbp"]

    print("final training setting:")
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

    ensure_dir("outputs/models")
    joblib.dump(model, "outputs/models/model.pkl")
    joblib.dump(class_names, "outputs/models/class_names.pkl")

    print("model saved to outputs/models/model.pkl")
    print("class names saved to outputs/models/class_names.pkl")


# Main Entry
def main(auto_search=False, search_mode="quick", save_results=True, plot_results=True):
    set_random_seed()

    best_setting = None
    results = None

    if auto_search:
        if search_mode == "quick":
            best_setting, results = quick_search_settings()
        elif search_mode == "full":
            best_setting, results = full_search_settings()
        else:
            raise ValueError("search_mode must be 'quick' or 'full'")

        if save_results and results is not None:
            save_search_results_csv(results)

        if plot_results and results is not None:
            plot_search_results(results)

    train_final_model(best_setting=best_setting)


if __name__ == "__main__":
    main(auto_search=True, search_mode="full")