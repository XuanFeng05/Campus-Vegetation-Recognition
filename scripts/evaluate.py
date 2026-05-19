import os
import json
import shutil
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from train_V3 import read_dataset, apply_setting

# def read_dataset(data_dir):
#     features=[]
#     labels = []
#     class_names=[]
#     filepaths = []#new, to find wrong result
#     for name in sorted(os.listdir(data_dir)):
#         if os.path.isdir(os.path.join(data_dir,name)):
#             class_names.append(name)
#     for classname in class_names:
#         path= os.path.join(data_dir,classname)

#         for filename in os.listdir(path):
#             filepath= os.path.join(path,filename)
#             image = cv2.imread(filepath)

#             if image is None:
#                 print(f"warn: can't read {filepath}!")
#                 continue

#             feature_color = extract_color(image)
#             feature_hog = extract_hog(image)
#             feature = np.hstack([feature_color,feature_hog])#connect two one-dimensional vectors
#             features.append(feature)
#             labels.append(classname)
#             filepaths.append(filepath)
#     return np.array(features),np.array(labels),filepaths,class_names

def main():
    test_dir="data/test"

    # Load the exact feature setting used in final training
    config_path = "outputs/models/active_config_v3.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("cannot find outputs/models/active_config_v3.json, please run train_V3.py first")

    with open(config_path, "r", encoding="utf-8") as f:
        best_setting = json.load(f)

    # Apply the saved setting before reading test data
    apply_setting(best_setting)

    print("load test data")
    features_test, labels_test, filepaths_test, class_names = read_dataset(
        test_dir,
        augment=False,
        return_filepaths=True
    )
    #class_names
    if len(features_test)==0:
        print("test set is empty")
        return

    print("load model")
    model=joblib.load("outputs/models/model.pkl")

    # Load saved class order for stable evaluation output
    saved_class_names = joblib.load("outputs/models/class_names.pkl")

    # Check whether feature dimension matches the trained model
    if hasattr(model, "n_features_in_"):
        if features_test.shape[1] != model.n_features_in_:
            raise ValueError(
                f"feature dimension mismatch: test={features_test.shape[1]}, model={model.n_features_in_}"
            )

    print("predicting")
    labels_pred = model.predict(features_test)

    print("wrong predictions")#show wrong predictions
    count = 0

    # Remove old wrong predictions to avoid mixing results
    wrong_root = "outputs/examples/wrong"
    if os.path.exists(wrong_root):
        shutil.rmtree(wrong_root)
    os.makedirs(wrong_root, exist_ok=True)

    for i in range(len(labels_test)):
        
        if labels_test[i]!=labels_pred[i]:
            true_class = labels_test[i]
            wrong_class_dir = os.path.join(wrong_root, true_class)
            os.makedirs(wrong_class_dir, exist_ok=True)
            print(filepaths_test[i],labels_test[i],labels_pred[i])
            count+=1
            if count<=50:
                filename = os.path.basename(filepaths_test[i])
                name, ext = os.path.splitext(filename)
                new_name = f"{name}_true_{labels_test[i]}_pred_{labels_pred[i]}{ext}"
                dst_path = os.path.join(wrong_class_dir, new_name)
                shutil.copy(filepaths_test[i], dst_path)
    print(f"There are {count} wrong predictions")

    acc = accuracy_score(labels_test,labels_pred)
    print(f"test accuracy is {acc:.3f}")

    cm = confusion_matrix(labels_test,labels_pred, labels=saved_class_names)
    print("confusion matrix:")
    print(cm)

    report = classification_report(labels_test,labels_pred, labels=saved_class_names)
    print("classification report:")
    print(report)

if __name__ =="__main__":
    main()