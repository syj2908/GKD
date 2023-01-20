import os
import torch
from GAF.thumos14.BDNet_student import BDNet_student
from GAF.common.config import config

"""
load part of pretrained params
"""

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
pretrained_path = config['training']['pretrained_path']
pretrained_path_cvae = config['training']['pretrained_path_cvae']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']

if __name__ == '__main__':
    net = BDNet_student(in_channels=config['model']['in_channels'],
                backbone_model=config['model']['backbone_model'])
    dict_trained = torch.load(pretrained_path)
    dict_new = net.state_dict().copy()

    new_list = list(net.state_dict().keys())
    trained_list = list(dict_trained.keys())
    print("new_state_dict size: {}  trained state_dict size: {}".format(
        len(new_list), len(trained_list)))
    print("New state_dict parameters names")
    print(new_list[:100])
    # print("trained state_dict parameters names")
    # print(trained_list)

    # print(type(dict_new))
    # print(type(dict_trained))