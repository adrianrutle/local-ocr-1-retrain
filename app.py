import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json

# CONFIGURATION
UPLOAD_FOLDER = 'data/images'
LABEL_FILE = 'data/labels.json'
MODEL_FOLDER = 'trocr_model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
if not os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, 'w') as f:
        json.dump([], f)

# Load processor and model (original or fine-tuned)
def load_model():
    model_path = MODEL_FOLDER if os.path.exists(os.path.join(MODEL_FOLDER, 'config.json')) else "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    return processor, model

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = None
    image_url = None
    filename = None
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + "." + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Run OCR
            processor, model = load_model()
            image = Image.open(filepath).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            image_url = url_for("uploaded_file", filename=filename)
            return render_template("index.html", transcription=transcription, image_url=image_url, filename=filename)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/correct", methods=["POST"])
def correct():
    corrected_text = request.form.get("corrected_text")
    filename = request.form.get("filename")
    if not filename or not corrected_text:
        flash("Correction failed.")
        return redirect(url_for("index"))
    # Save label pair
    with open(LABEL_FILE, "r+") as f:
        labels = json.load(f)
        labels.append({"image": filename, "text": corrected_text})
        f.seek(0)
        json.dump(labels, f, indent=2)
    flash("Correction saved! Ready for training.")
    return redirect(url_for("index"))

@app.route("/retrain", methods=["POST"])
def retrain():
    # Call retrain.py (subprocess for now)
    os.system("python retrain.py")
    flash("Retraining has been started! Check console for progress.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)