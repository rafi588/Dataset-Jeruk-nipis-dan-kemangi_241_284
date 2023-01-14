import os
import tensorflow as tf
import numpy as np
import skimage
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "model_tb.h5")

# Preprocess an image
def classify(model,image):
    class_names = ['Daun Jeruk', 'Daun Kemangi']
    new_image = plt.imread(image)
    resize_img = skimage.transform.resize(new_image, (150,150,3))
    pred = model.predict(np.array([resize_img]))
    list_index = [0,1]
    x = pred
    
    for i in range(2):
        for j in range(2):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    
    for i in range(1):
        label = class_names[list_index[i]]
        classified_prob = round(pred[0][list_index[i]] * 100)
    return label, classified_prob


# home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/klasifikasi")
def about():
    return render_template("klasifikasi.html")

@app.route("/klasifikasi", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("index.html")
    else:
        try:
            file = request.files["image"]
            upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(upload_image_path)
            file.save(upload_image_path)
        except FileNotFoundError:
            return render_template("index.html")    
        label,prob = classify(cnn_model, upload_image_path)
        prob = round((prob*100),2)
        
    return render_template(
        "klasifikasi.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/klasifikasi/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True
