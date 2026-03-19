# 🖼️ Image Captioning App

A Python-based web application that generates captions for images using deep learning.

## 🚀 Features

* Upload an image and generate captions
* Two modes:

  * 🧠 Custom CNN + LSTM model (trained from scratch)
  * ⚡ High-accuracy BLIP model
* Clean UI built with Flask + Bootstrap

## 🛠️ Tech Stack

* Python
* Flask
* PyTorch
* Transformers (BLIP)
* HTML/CSS (Bootstrap)

## 📦 Setup

```bash
git clone https://github.com/Anonymousperson05/image-captioning-app.git
cd image-captioning-app
pip install -r requirements.txt
python app.py
```

## 📁 Project Structure

```
├── app.py
├── utils.py
├── model/
│   ├── model.pth
│   └── vocab.pkl
├── templates/
│   └── index.html
├── train.py
```

## 🧠 Models

### Custom Model

* CNN Encoder (ResNet)
* LSTM Decoder

### BLIP Model

* Pretrained transformer model for high-quality captions

## 🎯 Usage

1. Upload an image
2. Select captioning mode
3. Generate caption

---

Built for academic project 🎓
