import torch
from model import MiniResNet34

def load_model(model_path):
    model = MiniResNet34()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model