# Handwritten Document Transcriber (with TrOCR)

A local Flask app to upload, transcribe, correct, and retrain TrOCR for your own handwriting.

## Features

- Upload JPEG/PNG handwritten document images
- Transcribes handwriting using TrOCR (Microsoft’s model)
- Lets you correct any mistakes
- Stores image-correction pairs for fine-tuning
- One-click retrain: updates TrOCR model with your corrections

## Setup

### 1. Native (local Python)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

### 2. Docker (recommended for easy setup)

1. **Build the Docker image**
   ```bash
   docker build -t local-ocr-app .
   ```
2. **Run the container**
   ```bash
   docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/trocr_model:/app/trocr_model local-ocr-app
   ```
   This will map your project’s data and trained model folders for persistence.

3. **Access the app**

   Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

- **Upload an image**: Choose a `.jpg`, `.jpeg`, or `.png` file with handwriting. Get instant OCR transcription. Correct the text if needed, then save.
- **Retrain the model**: Press "Retrain Model from Corrections" after you have saved a few corrections. This will fine-tune TrOCR (takes a few minutes if you have a GPU). All future transcriptions will use your personalized model.

## Notes

- Fine-tuning may run slowly on CPU; more training data = more time.
- Models are stored in the `trocr_model/` directory.
- You may run `retrain.py` alone or in a Jupyter notebook for more customization.

## Troubleshooting

- If retraining fails, make sure you've submitted at least one correction.
- To reset or backup your model, simply delete `trocr_model/`.
- For best accuracy, provide as many corrections as your patience allows.

## Credits

This tool uses [Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) via [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder).