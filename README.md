# Face Mask Detection 🎭

This project is a deep learning-based web application that detects whether a person is wearing a face mask or not, using a Convolutional Neural Network (CNN).

## 📌 Features
- Binary image classification: **Mask** or **No Mask**
- Trained with TensorFlow/Keras
- Simple web interface using Flask
- Lightweight model and fast predictions

## 🧠 Model
The model is trained on a dataset of face images with and without masks. It uses CNN layers and is saved as `mask_model.h5`.

You can find the training notebook here: [`Face Mask Detection.ipynb`](./Face%20Mask%20Detection.ipynb)

## 🚀 Web App (Flask)
Run the app with:

```bash
python app.py
