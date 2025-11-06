import os
import json
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from datasets import Dataset

MODEL_FOLDER = "trocr_model"
LABEL_FILE = "data/labels.json"
IMAGE_FOLDER = "data/images"

# 1. Prepare paired dataset from corrections
def load_dataset():
    with open(LABEL_FILE, "r") as f:
        pairs = json.load(f)
    images, texts = [], []
    for pair in pairs:
        img_path = os.path.join(IMAGE_FOLDER, pair["image"])
        if os.path.exists(img_path):
            images.append(img_path)
            texts.append(pair["text"])
    return images, texts

images, texts = load_dataset()
if not images or not texts:
    print("No training data found! Please correct transcriptions first.")
    exit(1)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    pixel = processor(image, return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids[0]
    return {"pixel_values": pixel, "labels": labels}

hf_data = [{"image": img, "text": txt} for img, txt in zip(images, texts)]
dataset = Dataset.from_list(hf_data)
dataset = dataset.map(preprocess)

training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=1,
    output_dir=MODEL_FOLDER,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=5,
    save_total_limit=1,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting fine-tuning on corrections...")
trainer.train()

model.save_pretrained(MODEL_FOLDER)
processor.save_pretrained(MODEL_FOLDER)
print("Fine-tuned model saved to", MODEL_FOLDER)