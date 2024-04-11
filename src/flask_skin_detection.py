from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

# Load the saved model
model_save_path = "C:\\Users\\VS70\\Desktop\\skin_disease\\src\\skin_ultra.h5"#Add your models path below to make the webapp work
model = load_model(model_save_path)

# Assuming you have a mapping of class indices to disease names
disease_mapping = {
    0: 'Bacterial Infections',
    1: 'Chronic skin conditions',
    2: 'Connective tissue disorder',
    3: 'Contact dermatitis',
    4: 'Fungal infections',
    5: 'Infestation and bites',
    6: 'Miscellaneous skin conditions',
    7: 'Nail conditions',
    8: 'Sexually transmitted infections',
    9: 'Viral infections'
}

def predict_disease(image_path):
    # Load the image and preprocess it
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the expected input shape of the model
    # img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    # print(predictions)
    

    # Get the predicted class index
    predicted_class = np.argmax(predictions)
    print(predicted_class)

    # Get the predicted disease name
    predicted_disease = disease_mapping.get(predicted_class, 'Unknown Disease')

    return predicted_disease

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        filename = secure_filename(file.filename)
        temp_image_path = os.path.join("src", "temp_image.jpg")
        file.save(temp_image_path)

        # Get the predicted disease
        predicted_disease = predict_disease(temp_image_path)

        return render_template('index.html', predicted_disease=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)
