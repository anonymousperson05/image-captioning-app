import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import nltk

class CaptionDataset(Dataset):
    def __init__(self, root, mapping, word2idx, transform):
        self.root = root
        self.data = []
        self.word2idx = word2idx
        self.transform = transform

        for img, caps in mapping.items():
            for cap in caps:
                self.data.append((img, cap))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]

        image_path = os.path.join(self.root, img_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())

        caption_idx = [self.word2idx["<start>"]]

        for tok in tokens:
            caption_idx.append(self.word2idx.get(tok, self.word2idx["<unk>"]))

        caption_idx.append(self.word2idx["<end>"])

        return image, torch.tensor(caption_idx)
        
def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded_captions = torch.zeros(len(captions), max_len).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]

    return images, padded_captions