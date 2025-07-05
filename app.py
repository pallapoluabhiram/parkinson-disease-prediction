from flask import Flask, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import lime
import lime.lime_image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
import locale
sys.stdout.reconfigure(encoding='utf-8')

# Set system encoding to UTF-8
if locale.getpreferredencoding() != 'UTF-8':
    os.environ['PYTHONUTF8'] = '1'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['EXPLANATIONS_FOLDER'] = 'static/explanations'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXPLANATIONS_FOLDER'], exist_ok=True)

# Load the model once globally
try:
    model = load_model('model_1.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Function to get detailed info based on prediction
def get_detailed_info(prediction_result):
    if prediction_result == "Parkinson's Detected":
        return """
        <h5>Stages of Parkinson's Disease</h5>
        <ul>
            <li><strong>Stage 1 (Early Symptoms):</strong> 
                <ul>
                    <li>Mild tremors</li>
                    <li>Facial expression changes</li>
                    <li>Slight movement issues on one side of the body</li>
                </ul>
            </li>
            <li><strong>Stage 2:</strong> 
                <ul>
                    <li>Symptoms on both sides of the body</li>
                    <li>Walking and posture problems</li>
                    <li>Muscle stiffness</li>
                </ul>
            </li>
            <li><strong>Stage 3 (Mid-Stage):</strong> 
                <ul>
                    <li>Balance issues</li>
                    <li>Falls are more common</li>
                    <li>Slowness of movement</li>
                </ul>
            </li>
            <li><strong>Stage 4:</strong> 
                <ul>
                    <li>Severe disability</li>
                    <li>Assistance needed for daily activities</li>
                    <li>Trouble standing without help</li>
                </ul>
            </li>
            <li><strong>Stage 5 (Advanced Stage):</strong> 
                <ul>
                    <li>Wheelchair or bedridden</li>
                    <li>Delusions or hallucinations</li>
                    <li>Requires full-time care</li>
                </ul>
            </li>
        </ul>
        <h6>Common Non-Motor Symptoms:</h6>
        <ul>
            <li>Depression</li>
            <li>Sleep problems</li>
            <li>Fatigue</li>
            <li>Constipation</li>
            <li>Loss of smell</li>
        </ul>
        <p><strong>Consult a neurologist for a comprehensive diagnosis and care plan.</strong></p>
        """
    else:
        return """
        <h5>No Signs of Parkinson's Disease Detected</h5>
        <ul>
            <li>Your brain scan appears normal.</li>
            <li>However, if you have symptoms, please consult a neurologist for professional guidance.</li>
        </ul>
        """

# Function to generate LIME explanation
def generate_lime_explanation(img_path, model):
    explainer = lime.lime_image.LimeImageExplainer()
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    def predict_fn(images):
        return model.predict(images)

    explanation = explainer.explain_instance(img_array[0], predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
    fig, ax = plt.subplots()
    ax.imshow(temp)
    ax.imshow(mask, cmap='jet', alpha=0.5)
    ax.axis('off')

    lime_img_path = os.path.join(app.config['EXPLANATIONS_FOLDER'], 'lime_explanation.png')
    plt.savefig(lime_img_path, bbox_inches='tight')
    plt.close()
    
    return lime_img_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

    try:
        prediction = model.predict(img_array)
        confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
        result = "Parkinson's Detected" if prediction[0][0] > 0.5 else "Normal"

        # Get info + explanation
        detailed_info = get_detailed_info(result)
        lime_img_path = generate_lime_explanation(filepath, model)

    except Exception as e:
        return f"Error making prediction: {str(e)}", 500

    return render_template(
        'result.html',
        result=result,
        confidence=f"{confidence:.2f}%",
        image_url=url_for('static', filename=f'uploads/{file.filename}'),
        detailed_info=detailed_info,
        lime_explanation=url_for('static', filename='explanations/lime_explanation.png')
    )

if __name__ == '__main__':
    app.run(debug=True)
