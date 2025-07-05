🧠 Project Title: Parkinson's Disease Detection with MRI Scans using Deep Learning & LIME
🧬 Project Overview:
This is a web-based AI application built using Flask, which detects Parkinson’s Disease from MRI brain scan images using a pre-trained deep learning model. The app also includes LIME (Local Interpretable Model-agnostic Explanations) to visually explain the model's prediction, making it more interpretable for users and medical professionals.

🧩 Technologies Used:
Category	Tools & Libraries
Frontend	HTML, CSS, Bootstrap (via Jinja templates)
Backend	Flask, Python
Deep Learning	TensorFlow, Keras
Image Processing	Pillow (PIL), OpenCV, Matplotlib
Model Explainability	LIME (lime-image)
Utilities	NumPy, Werkzeug

🧪 Key Features:
🧠 Parkinson's Prediction:
Accepts an MRI image uploaded by the user.

Preprocesses the image (resizing, normalization).

Uses a pre-trained deep learning model (model_1.h5) to classify the image as:

✅ "Normal"

❌ "Parkinson's Detected"

Displays confidence score for each prediction.

🧾 Dynamic Medical Info:
Shows stage-wise progression of Parkinson’s Disease if detected.

Displays precautionary messages and neurological advice for both outcomes.

🔍 LIME Interpretability:
Uses LIME (Local Interpretable Model-Agnostic Explanations) to generate a visual explanation (heatmap overlay).

Highlights which parts of the MRI scan influenced the model's decision.

Explanation image saved and served from static/explanations/lime_explanation.png.

📁 Project Structure (Overview):
swift
Copy
Edit
/static/
    /uploads/ → stores uploaded MRI images
    /explanations/ → stores generated LIME explanation images
/templates/
    index.html → Home page (file upload)
    result.html → Displays result, confidence, LIME visualization
app.py → Main Flask app
model_1.h5 → Pre-trained CNN model
⚙️ How It Works:
User uploads MRI image via a web form (index.html).

Image is saved, resized to (224x224), normalized, and passed to the model.

Prediction is made: Parkinson’s or Normal.

A confidence score is calculated and displayed.

A LIME explanation image is generated, highlighting areas of interest in the MRI.

The result page shows:

Prediction

Confidence %

Uploaded MRI image

LIME explanation overlay

Stage-wise disease info (if applicable)

🩺 Use Case:
This application is designed for early detection and awareness of Parkinson’s Disease from medical scans. It provides both:

A prediction via deep learning, and

Visual, interpretable evidence using LIME

This helps both patients and clinicians gain trust in AI-based tools.

🚀 Future Improvements:
Add Grad-CAM or SHAP for deeper visualization.

Support multiple medical formats (e.g., DICOM).

Integrate with Electronic Health Records (EHR).

Deploy to cloud with real-time inference API.
