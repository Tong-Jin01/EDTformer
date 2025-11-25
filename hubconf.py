dependencies = ['torch']

import torch
import network

def EDTformer():
    model = network.VPRNet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(f'https://github.com/Tong-Jin01/EDTformer/releases/download/v1.0.0/EDTformer.pth', map_location=torch.device('cpu'))["model_state_dict"]
    )
    return model
