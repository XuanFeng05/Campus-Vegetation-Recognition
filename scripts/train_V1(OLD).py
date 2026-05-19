import os
import cv2
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
import csv

from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
from skimage.feature import hog, local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# final best setting
IMG_SIZE = (128, 128)

HOG_PIXELS_PER_CELL = (24, 24)
HOG_ORIENTATIONS = 9
USE_LBP = True
COLOR_BINS = (16, 16, 16)

RANDOM_SEED = 4423


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def extract_color(image):
    """extract image color histogram"""
    image = cv2.resize(image, IMG_SIZE)
    hist = cv2.calcHist(
        [image], [0, 1, 2], None,
        COLOR_BINS,
        [0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_hog(image):
    """HOG features <- grayscale"""
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray the image
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=(2, 2),
        feature_vector=True  # return a one-dimensional vector
    )
    return features.astype(np.float32)


def extract_canny(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


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
    """combine features"""
    # old tested versions
    # feature_color = extract_color(image)
    # feature_hog = extract_hog(image)
    # feature_lbp = extract_lbp(image)
    # feature_canny = extract_canny(image)
    # feature = np.hstack([
    #     feature_color,
    #     feature_hog,
    #     # feature_lbp,
    #     # feature_canny
    # ]).astype(np.float32)

    feature_color = extract_color(image)
    feature_hog = extract_hog(image)

    if USE_LBP:
        feature_lbp = extract_lbp(image)
        feature = np.hstack([feature_color, feature_hog, feature_lbp]).astype(np.float32)
    else:
        feature = np.hstack([feature_color, feature_hog]).astype(np.float32)

    return feature


def save_search_results_csv(results, save_path="results/feature_search_results.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x_labels = []
    seen = set()
    for row in results:
        label = f'{row["pixels_per_cell"][0]}x{row["pixels_per_cell"][1]} | ori={row["orientations"]}'
        if label not in seen:
            seen.add(label)
            x_labels.append(label)

    group_dict = {}
    for row in results:
        group_name = f'bins={row["color_bins"][0]}, lbp={row["use_lbp"]}'
        x_label = f'{row["pixels_per_cell"][0]}x{row["pixels_per_cell"][1]} | ori={row["orientations"]}'
        if group_name not in group_dict:
            group_dict[group_name] = {}
        group_dict[group_name][x_label] = row["best_cv_accuracy"]

    plt.figure(figsize=(12, 6))

    for group_name, mapping in group_dict.items():
        y_values = [mapping.get(x, np.nan) for x in x_labels]
        plt.plot(x_labels, y_values, marker="o", label=group_name)

    plt.xlabel("HOG setting")
    plt.ylabel("Best 5-fold CV accuracy")
    plt.title("Feature search results")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"plot saved to: {save_path}")


def search_feature_settings():
    global HOG_PIXELS_PER_CELL, HOG_ORIENTATIONS, USE_LBP, COLOR_BINS

    train_dir = "data/train"

    color_bin_options = [(8, 8, 8), (16, 16, 16)]
    pixel_options = [(16, 16), (24, 24), (32, 32), (40, 40)]
    orientation_options = [9, 12, 15, 18]
    lbp_options = [False, True]

    best_score = -1
    best_setting = None
    results = []

    for color_bins in color_bin_options:
        for pixels in pixel_options:
            for ori in orientation_options:
                for use_lbp in lbp_options:
                    COLOR_BINS = color_bins
                    HOG_PIXELS_PER_CELL = pixels
                    HOG_ORIENTATIONS = ori
                    USE_LBP = use_lbp

                    print("\n==============================")
                    print("testing setting:")
                    print("color_bins =", COLOR_BINS)
                    print("pixels_per_cell =", HOG_PIXELS_PER_CELL)
                    print("orientations =", HOG_ORIENTATIONS)
                    print("use_lbp =", USE_LBP)

                    features_train, labels_train, _ = read_dataset(train_dir, augment=False)

                    print("train samples:", len(features_train))
                    print("feature shape:", features_train.shape)

                    search = build_search_model()
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

    save_search_results_csv(results)
    plot_search_results(results)

    return best_setting, results

def read_dataset(data_dir, augment=False):
    """read the dataset"""
    features = []  # features -> X
    labels = []  # -> y
    class_names = []  # class names
    # class_names = sorted(os.listdir(data_dir))

    if not os.path.exists(data_dir):
        print(f"can not found the directory {data_dir}")
        return np.array(features, dtype=np.float32), np.array(labels), class_names

    for name in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, name)):
            class_names.append(name)

    for classname in class_names:  # read all classes' name
        path = os.path.join(data_dir, classname)
        # if not os.path.isdir(path):
        #     continue
        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)

            if not os.path.isfile(filepath):
                continue
            if os.path.splitext(filename)[1].lower() not in VALID_EXTENSIONS:  # if valid extension
                continue
            image = cv2.imread(filepath)  # read images
            if image is None:
                print(f"warn: can't read {filepath}!")
                continue

            if augment:
                image_list = gen_aug_imag(image)
            else:
                image_list = [image]

            for img in image_list:
                feature = extract_features(img)  # extract features
                features.append(feature)
                labels.append(classname)

    return np.array(features, dtype=np.float32), np.array(labels), class_names


# old build_model
# def build_model(modelname):
#     if modelname=="svm":
#         return Pipeline([
#             ("standardize",StandardScaler()),
#             ("classifier",SVC(kernel="rbf",C=1.0,gamma="scale"))
#         ])
#     else:
#         raise ValueError("unsupported model")


def build_search_model():
    pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ])
    param_grid = {
        "classifier__C": [0.1, 1, 10, 50],
        "classifier__gamma": ["scale", 0.01, 0.001, 0.0001]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
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


def gen_aug_imag(image):
    augmented = [
        image,                     # original
        rotate_image(image, 15),   # +15 degree
        rotate_image(image, -15),  # -15 degree
        cv2.flip(image, 1),        # horizontal flip
        adjust_brightness(image, 0.8),  # darker
        adjust_brightness(image, 1.2)   # brighter
    ]
    return augmented


def main(modelname="svm"):
    set_random_seed()

    train_dir = "data/train"

    print("now load training data")
    features_train, labels_train, class_names = read_dataset(train_dir, augment=False)

    if len(features_train) == 0:
        print("training set is empty")
        return

    print("train samples:", len(features_train))
    print("feature shape:", features_train.shape)

    print("start 5-fold cross-validation")
    search = build_search_model()
    search.fit(features_train, labels_train)
    print(f"best cv accuracy:{search.best_score_:.4f}")
    print("best parameters:", search.best_params_)

    print("now load augmented training data for final training")
    features_train_aug, labels_train_aug, _ = read_dataset(train_dir, augment=True)

    print("augmented train samples:", len(features_train_aug))
    print("augmented feature shape:", features_train_aug.shape)

    print("now train final model on full training set")
    model = build_final_model(search.best_params_)
    model.fit(features_train_aug, labels_train_aug)

    # return

    # store results
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/model.pkl")
    joblib.dump(class_names, "outputs/models/class_names.pkl")
    print("model and class names have been saved.")


if __name__ == "__main__":
    main()
    # set_random_seed()
    # search_feature_settings()