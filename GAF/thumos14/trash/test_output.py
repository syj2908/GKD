import torch
from torchsummary import summary
from GAF.thumos14.BDNet import BDNet
from GAF.common.config import config

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BDNet(in_channels=config['model']['in_channels'],
                backbone_model=config['model']['backbone_model']).to(device)
    summary(net, (3, 256, 96, 96))
