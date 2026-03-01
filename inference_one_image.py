import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from PIL import Image
from Config  import OPML_config
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from dataset import myDataset
from util import *

from OPML import OPML_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg, args):
    model = create_model(pretrained=False, cfg=cfg).to(cfg.device)
    checkpoint = torch.load(args.load_ckpt_path, map_location=cfg.device)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
    print("weights had been load!\n")

    transform= transforms.Compose([
            ransforms.Resize((512, 512),interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    transform1 = transforms.Compose([
            transforms.Resize((224, 224),interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    image = Image.open(args.test_img_path)
    image1 = transform(image)
    image2 = transform1(image)
    text1 ='This is a good image'
    text2 ='This is a Poor image'
    text3 ='This is a Fair image'
    text4 ='This is a excellent image'


    model.eval()
    pred = model(image1.to(device),image2.to(device),text1,text2,text3,text4)
    print('Pred Score: {}'.format(pred))
        
if __name__ == '__main__':
    cfg = OPML_config()
    parse = argparse.ArgumentParser()

    parse.add_argument('--load_ckpt_path', type=str, default='/home/xxxy/Zjy/OPML/ckpt/best_epoch.pth')
    parse.add_argument('--test_img_path', type=str, default='/home/xxxy/Zjy/databases/dis535.png')
    parse.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    args = parse.parse_args()

    main(cfg, args)