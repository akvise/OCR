import config
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset import TextDataset


def decode_predictions(text_batch_logits):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2)  # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T  # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [config.LABEL2CHAR[int(idx)] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


def real_text(text_tokens):
    text_tokens = text_tokens.numpy()
    tr = []
    for text_token in text_tokens:
        text = [config.LABEL2CHAR[int(idx)] for idx in text_token]
        text = "".join(text)
        tr.append(text)
    return tr


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)

def correct_prediction(word):
    parts = word.split(" ")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def make_prediction(model, path: str, imgs: list) -> list[str]:
    dataset = TextDataset(path, imgs, test_mode=True)
    dataloader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        pred = []
        for image_batch, text_batch in dataloader:
            text_batch_logits = model(image_batch.to(config.device))  # [T, batch_size, num_classes==num_features]
            text_batch_pred = decode_predictions(text_batch_logits.cpu())
            pred += text_batch_pred
    pred = [correct_prediction(t) for t in pred]
    return pred

