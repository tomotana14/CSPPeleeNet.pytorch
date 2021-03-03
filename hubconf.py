dependencies = ['torch']
import torch
from peleenet import PeleeNet


state_dict_url = 'https://github.com/tomotana14/CSPPeleeNet.pytorch/raw/master/weights/csppeleenet.pth'

def csppeleenet(pretrained=False, **kwargs):
    net = PeleeNet(partial_ratio=0.5)
    if pretrained:
        if torch.cuda.is_available() : 
            state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True)
        else:
            state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True, map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)
    return net
