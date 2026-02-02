import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import csv
Image.MAX_IMAGE_PIXELS=None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

class myDataset(Dataset):
    def __init__(self, cfg ,transform,transform1,csv_path):
        self.root_dir = cfg.root_dir
        self.transform1 = transform
        self.transform2 = transform1
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)

        # column_names =   ['ref','name', 'mos','text1','text2','text3','text4']
        column_names =   ['name','stage', 'score','mos','mean','text1','text2','text3','text4']
        self.df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.image_name = self.df['name']
        self.mos=self.df['mos']
        self.text1=self.df['text1']
        self.text2=self.df['text2']
        self.text3=self.df['text3']
        self.text4=self.df['text4']
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,index):

        image_path = os.path.join(self.root_dir, self.image_name[index])


        image = Image.open(image_path).convert('RGB')

        label = torch.FloatTensor(np.array(self.mos[index]))
        text1=self.text1[index]
        text2=self.text2[index]
        text3=self.text3[index]
        text4=self.text4[index]
        

        image1 = self.transform1(image)
        image2 = self.transform2(image)
        


        return image1,image2,label,text1,text2,text3,text4

    def get_label_from_csv(self, image_file):

        pass
    
    