import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import clip
import torch.nn.functional as F
from longclip_model import longclip
device='cuda'

def find_max(a, b, c, d):
    max_value = 0
    max_variable = 'similarity1'
    max_variable_list = []
    for i in range(8):
        if a[i] > max_value:
            max_value = a[i]
            max_variable = 'similarity1'
        if b[i] > max_value:
            max_value = b[i]
            max_variable = 'similarity2'
        if c[i] > max_value:
            max_value = c[i]
            max_variable = 'similarity3'
        if d[i] > max_value:
            max_value = d[i]
            max_variable = 'similarity4'
        max_variable_list.append(max_variable)
        max_value=0
    return max_variable_list

class max_textencode(nn.Module):  

    def __init__(self):
        super().__init__()
        self.model, preprocess = longclip.load("/mnt/10T/zjy/D_OIQA/longclip_model/longclip-B.pt", device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.my_learnable_tensor1 = nn.Parameter(torch.randn(2048))
        self.my_learnable_tensor = nn.Parameter(torch.randn(512))
        self.FC=nn.Linear(2048,4)
        self.relu=nn.ReLU()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        
    def forward(self,x,text1,text2,text3,text4):
        texts1=longclip.tokenize(text1)
        texts2=longclip.tokenize(text2)
        texts3=longclip.tokenize(text3)
        texts4=longclip.tokenize(text4)
        texts1=texts1.to(device)
        texts2=texts2.to(device)
        texts3=texts3.to(device)
        texts4=texts4.to(device)
        text_features1=self.model.encode_text(texts1)
        text_features2=self.model.encode_text(texts2)
        text_features3=self.model.encode_text(texts3)
        text_features4=self.model.encode_text(texts4)
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        text_features3 = text_features3 / text_features3.norm(dim=-1, keepdim=True)
        text_features4 = text_features4 / text_features4.norm(dim=-1, keepdim=True)
        image_features=self.model.encode_image(x)
        image_features=image_features/image_features.norm(dim=1,keepdim=True)
        similarity1 = (self.logit_scale*image_features @ text_features1.T).softmax(dim=-1)

        similarity1=torch.diag(similarity1)


        
        similarity2 = (self.logit_scale*image_features @ text_features2.T).softmax(dim=-1)
        similarity2=torch.diag(similarity2)


        
        similarity3 = (self.logit_scale*image_features @ text_features3.T).softmax(dim=-1)
        similarity3=torch.diag(similarity3)

        
        similarity4 = (self.logit_scale*image_features @ text_features4.T).softmax(dim=-1)
        similarity4=torch.diag(similarity4)


        max_similarity_list = find_max(similarity1,similarity2,similarity3,similarity4) 

        
        for i in range(8):
            if max_similarity_list[i]=='similarity1':
                text_features_list.append(text_features1[i])
            if max_similarity_list[i]=='similarity2':
                text_features_list.append(text_features2[i])
            if max_similarity_list[i]=='similarity3':
                text_features_list.append(text_features3[i])
            if max_similarity_list[i]=='similarity4':
                text_features_list.append(text_features4[i])
    
        text_feature = torch.stack(text_features_list)
      
       

        return probs,text_features1,text_features2,text_features3,text_features4,image_features,text_feature

class DME(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, 1, kernel_size, stride=1,padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes//2, kernel_size, stride=1,padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_planes//2, in_planes, kernel_size, stride=1,padding=0, bias=True)
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avg_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        
        self.max_conv = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        
        
        self.gate1 = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes, 1, bias=False),
            nn.Sigmoid()
        )


        
    def forward(self, x):
        # x:  [batch_size, channels, height, width]
        x2=x
        x1=x
        b,c,w,h=x.size()
        x=self.conv1(x)

        x1=x1.view(b,c,h*w).unsqueeze(1)#[b,1,c,h*w]
        x=x.view(b,1,h*w)
        x=F.softmax(x,dim=-1).unsqueeze(-1)#[b,1,h*w,1]
        x1_x=torch.matmul(x1,x).view(b,c,1,1)
        
        x1_x=self.conv3(self.relu(self.conv2(x1_x)))
        x1_x=x1_x+x2


        avg_out = self.avg_conv(x1_x)
        max_out = self.max_conv(x1_x)
        
        
        out = torch.cat([avg_out, max_out], dim=1)
        
        
        out = self.gate1(out)
 
        
        return out*x1_x

  

class AlternatingFusionModel(nn.Module):
    def __init__(self, fusion_dim=2560):
        super(AlternatingFusionModel, self).__init__()

        self.fusion_layer = nn.Linear(2560, 512)
        
    def forward(self, text, image):

        
        combined_features = torch.cat((text, image), dim=1)
        text_fused_features = torch.tanh(self.fusion_layer(combined_features))+text
        

        
        return text_fused_features


class OPML_model(nn.Module):
    def __init__(self, num_classes=1):
        super(PQ_clip_Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(self.model.children())[:-1])
        self.pre=nn.ModuleList([self.model.conv1, self.model.bn1, 
                                       self.model.relu, self.model.maxpool])
        self.backbone = nn.ModuleList([self.model.layer1, self.model.layer2, 
                                       self.model.layer3, self.model.layer4])
        self.AFM = AlternatingFusionModel()
        self.DME_4 = SpatialAttention(2048,2048)
        self.DME_3 = SpatialAttention(1024,1024)
        self.DME_2 = SpatialAttention(512,512)
        self.DME_1 = SpatialAttention(256,256)
        self.dropout = nn.Dropout(0.5)  
        self.fc=nn.Linear(2048,512)
        self.bn4 = nn.BatchNorm2d(2048)
        self.conv1 = nn.Conv2d(in_channels=5120, out_channels=2048, kernel_size=1, stride=1, padding=0)
        self.conv_4_3 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.conv_4_2 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_4_1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv_3_2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_3_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv_2_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu=nn.ReLU()
        self.reduce_channels = nn.Conv2d(3840, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.Avgpool2d=nn.AdaptiveAvgPool2d((1, 1))
        self.max_textencode=max_textencode()
        self.classifier =  nn.Sequential(

            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Linear(512,num_classes)  
        )

    def HVDM(self,feature_maps):
        fused_feature = None
        feature_map1=feature_maps[0]
        feature_map2=feature_maps[1]
        feature_map3=feature_maps[2]
        feature_map4=feature_maps[3]
        
        
        feature_map4_3=F.interpolate(feature_map4, size=feature_map3.size()[2:], mode='bilinear', align_corners=False)#这里代表  第几层往第几层下采样
        feature_map4_2=F.interpolate(feature_map4, size=feature_map2.size()[2:], mode='bilinear', align_corners=False)
        feature_map4_1=F.interpolate(feature_map4, size=feature_map1.size()[2:], mode='bilinear', align_corners=False)
        feature_map4_3=self.conv_4_3(feature_map4_3)
        feature_map4_2=self.conv_4_2(feature_map4_2)
        feature_map4_1=self.conv_4_1(feature_map4_1)
    
    
        feature_map3_2=F.interpolate(feature_map3, size=feature_map2.size()[2:], mode='bilinear', align_corners=False)
        feature_map3_1=F.interpolate(feature_map3, size=feature_map1.size()[2:], mode='bilinear', align_corners=False)
        feature_map3_2=self.conv_3_2(feature_map3_2)
        feature_map3_1=self.conv_3_1(feature_map3_1)
        
        feature_map2_1=F.interpolate(feature_map2, size=feature_map1.size()[2:], mode='bilinear', align_corners=False)
        feature_map2_1=self.conv_2_1(feature_map2_1)
         
        feature_map1_4_3_2=feature_map1*feature_map4_1*feature_map3_1*feature_map2_1

        feature_map1_4_3_2=self.DME_1(feature_map1_4_3_2)

        feature_map2_4_3=feature_map2*feature_map3_2*feature_map4_2
        feature_map2_4_3=self.DME_2(feature_map2_4_3)

        feature_map3_4=feature_map3*feature_map4_3 
        feature_map3_4=self.DME_3(feature_map3_4)

        
        feature_map4=self.DME_4(feature_map4)

        
        feature_maps_plus=[feature_map1_4_3_2,feature_map2_4_3,feature_map3_4,feature_map4]  
                
        for feature_map in reversed(feature_maps_plus): 
            if fused_feature is None:
                fused_feature = feature_map
            else:
 
                fused_feature = F.interpolate(fused_feature, size=feature_map.size()[2:], mode='bilinear', align_corners=False)
 
                fused_feature = torch.cat((fused_feature, feature_map), dim=1)
        return fused_feature

    
    def forward(self,image1,image2,text1,text2,text3,text4):
        probs,text_features1,text_features2,text_features3,text_features4,image,text_features=self.max_textencode(image2,text1,text2,text3,text4)
       
        text_feature = text_features
        feature_map=[]
        x=image1
        for pre in self.pre:
            x=pre(x)
        for layer in self.backbone:
            x=layer(x)
            feature_map.append(x)
        fused_feature=self.HVDM(feature_map)
        x=self.Avgpool2d(fused_feature)
        x=self.reduce_channels(x)
        x = x.view(x.size(0), -1)
        text_fusion_feature = self.AFM(text_feature,x)
        i_t=torch.cat((x,text_fusion_feature),dim=1)
        i_t=self.classifier(i_t)
        return i_t
