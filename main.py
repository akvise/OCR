import config
from model.model import CRNN
import torch

from predict.predict import make_prediction

if __name__ == '__main__':
    model = CRNN(len(config.vocabs))
    model.load_state_dict(torch.load('checkpoint/checkpoint.pth', map_location=torch.device('cpu')))

    print(make_prediction(model, 'img', ['img1.png']))






