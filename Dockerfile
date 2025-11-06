# Use official Python base image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py retrain.py ./
COPY templates ./templates

RUN mkdir -p data/images trocr_model

ENV FLASK_RUN_PORT=5000
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]