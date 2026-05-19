import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import joblib

from train import extract_color,extract_hog
import os
BASE_DIR=os.path.dirname(__file__)
PROJECT_DIR=os.path.dirname(BASE_DIR)
MODEL_PATH=os.path.join(PROJECT_DIR,"outputs","models","model.pkl")


model = joblib.load(MODEL_PATH)
    
    
def choose_image():
    filepath = filedialog.askopenfilename()
    if filepath=="":return
    image = cv2.imread(filepath)
    if image is None:
        result_label.config(text= "cannot read image")
        return
    

    feature_color = extract_color(image)
    feature_hog = extract_hog(image)
    feature = np.hstack([feature_color,feature_hog])
    feature = feature.reshape(1,-1)
    pred_label = model.predict(feature)[0]
    result_label.config(text = f"predicted class:{pred_label}")


window= tk.Tk()
window.title("campus vegetation recognition")

button = tk.Button(window,text="choose image",command=choose_image)
button.pack()

result_label = tk.Label(window,text="prediction will appear here")
result_label.pack()

window.mainloop()




