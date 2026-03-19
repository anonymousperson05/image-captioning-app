import nltk
from collections import Counter

nltk.download('punkt')

def build_vocab(captions, threshold=5):
    counter = Counter()

    for cap in captions:
        tokens = nltk.tokenize.word_tokenize(cap.lower())
        counter.update(tokens)

    words = [w for w, cnt in counter.items() if cnt >= threshold]

    word2idx = {w: i+4 for i, w in enumerate(words)}
    word2idx["<pad>"] = 0
    word2idx["<start>"] = 1
    word2idx["<end>"] = 2
    word2idx["<unk>"] = 3

    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word
    