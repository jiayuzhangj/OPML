import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from PIL import Image
from Config  import OPML_config
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


def main(cfg):
    set_seed(cfg)
    # create model
    print("*****begin test*******************************************************")
    model = create_model(pretrained=False, cfg=cfg).to(cfg.device)
    checkpoint = torch.load("XXX/XXX/XXX/best_epoch.pth", map_location=cfg.device)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
    print("weights had been load!\n")
   test_dataset = dataset(
        cfg = OPML_config(),
        csv_path = cfg.test_csv_path,
        transform = transforms.Compose([
          
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
transform1 = transforms.Compose([
            transforms.Resize((224, 224),interpolation=BICUBIC),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,drop_last=True, num_workers=cfg.num_workers,shuffle=True)

    # test
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        # for image1,image2,labels,text in tqdm(test_loader):
        for image1,image2,labels,text1,text2,text3,text4 in tqdm(test_loader):
            pred = 0
          
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
           
            pred = net(image1.to(device),image2.to(device),text1,text2,text3,text4)
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = mean_squared_error(np.squeeze(pred_epoch),np.squeeze(labels_epoch),squared=False)
            
        print('loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ==== RMSE{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p,rmse))

        
if __name__ == '__main__':
    cfg = OPML_config()
    main(cfg)