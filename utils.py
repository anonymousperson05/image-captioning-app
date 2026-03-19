from __future__ import annotations

import base64
from io import BytesIO
from typing import Tuple, Any
import pickle

import torch
from PIL import Image
from torchvision import transforms

from model import Model

MODEL_PATH = "model/model.pth"
VOCAB_PATH = "model/vocab.pkl"

from transformers import BlipProcessor, BlipForConditionalGeneration

# load BLIP once
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

blip_model.eval()

def generate_caption_blip(image: Image.Image):
    device = next(blip_model.parameters()).device

    inputs = blip_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(**inputs)

    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


def load_model(device_override: str | None = None):
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))

    # load vocab
    with open(VOCAB_PATH, "rb") as f:
        word2idx, idx2word = pickle.load(f)

    #  rebuild model
    model = Model(256, 512, len(word2idx)).to(device)

    #  load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    return model, device, word2idx, idx2word

# same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess_image(image: Image.Image, device: torch.device):
    image = transform(image).unsqueeze(0)
    return image.to(device)


def generate_caption(model, image_tensor, word2idx, idx2word, max_len=30, beam_size=3):
    model.eval()

    start_token = word2idx["<start>"]
    end_token = word2idx["<end>"]

    with torch.no_grad():
        features = model.encoder(image_tensor)

        sequences = [[ [start_token], 0.0 ]]

        for _ in range(max_len):
            all_candidates = []

            for seq, score in sequences:
                if seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue

                inputs = torch.tensor(seq).unsqueeze(0).to(image_tensor.device)
                outputs = model.decoder(features, inputs)
                outputs = outputs[:, -1, :]

                probs = torch.log_softmax(outputs, dim=-1)
                top_probs, top_idx = probs.topk(beam_size)

                for i in range(beam_size):
                    candidate = [
                        seq + [top_idx[0][i].item()],
                        score + top_probs[0][i].item()
                    ]
                    all_candidates.append(candidate)

            # pick best sequences
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_size]

        best_seq = sequences[0][0]

    words = [idx2word[idx] for idx in best_seq]

    return " ".join(words[1:-1])


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")