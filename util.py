import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from tqdm import tqdm
import logging
import numpy as np
from scipy.stats import spearmanr, pearsonr
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_epoch(epoch, model, criterion, optimizer, scheduler, train_loader):
    losses = []
    model.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for img,labels in tqdm(train_loader):
        image=img.cuda()
        image=image.permute(0,2,1,3,4)
        # print(image.device)
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        pred_d = model(image)
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        # print(pred_epoch)
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p

def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for image,labels in tqdm(test_loader):
            pred = 0
            image=image.permute(0,2,1,3,4)
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            pred = net(image.to(device))


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

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p))
        return np.mean(losses), rho_s, rho_p

def mean_squared_error(actual, predicted, squared=True):
    """MSE or RMSE (squared=False)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    error = predicted - actual
    res = np.mean(error**2)
    
    if squared==False:
        res = np.sqrt(res)
    
    return res

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
def cf_lb(name,labels):
    cf_label=[]
    if name=='JUFE-10k':
        for i in range(8):
            if labels[i]<2.0:
                cf_label.append(0)#0 denotes poor
            elif labels[i]>=2.0 and labels[i]<3.0:
                cf_label.append(1)#1 denotes fair
            else:
                cf_label.append(2)#2 denotes good
    if name=='OIQ-10k':
        for i in range(8):
            if labels[i]<2.0:
                cf_label.append(0)#0 denotes poor
            else:
                cf_label.append(1)#1 denoets good
    return cf_label
            
def train_clip_epoch(epoch, model, criterion,criterion_choose, optimizer, scheduler, train_loader):
    losses = []
    model.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    # for img1,img2,labels,text in tqdm(train_loader):
    for img1,img2,labels,text1,text2,text3,text4 in tqdm(train_loader):
        image1=img1.cuda()
        image2=img2.cuda()
        # image3=img3.cuda()
        # position=position.cuda()
        # print(image.device)
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        
        # texts="This is a {} image"
        # texts=texts.format(text)
        # texts = ["This is a {} image".format(t) for t in text]
        # texts = "\n".join(texts)  # Join the texts into a single string separated by newlines
        # pred_d,probs = model(image1,image2,text1)
        pred_d = model(image1,image2,text1,text2,text3,text4)
        optimizer.zero_grad()
        cf_label = cf_lb('JUFE-10k',labels)
        cf_label = torch.tensor(cf_label).to(device=device).float()
        loss1 = criterion(torch.squeeze(pred_d), labels)
        # loss2 = criterion_choose(torch.squeeze(probs),cf_label)
        # print(loss1)
        # print(loss2)
        # print(probs)
        # print(cf_label)
        loss = loss1#+loss2
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        # print(pred_epoch)
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = mean_squared_error(np.squeeze(pred_epoch),np.squeeze(labels_epoch),squared=False)

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p,rmse))

    return ret_loss, rho_s, rho_p


def eval_clip_epoch(config, epoch, net, criterion,criterion_choose, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        # for image1,image2,labels,text in tqdm(test_loader):
        for image1,image2,labels,text1,text2,text3,text4 in tqdm(test_loader):
            pred = 0
            # texts="Is this a Good image or a Poor image?"
            #  # Expand the string to the batch size
            # batch_size = image.size(0)  # Get the batch size
            # texts = [texts] * batch_size  # Create a list where each element is the string
            # texts=texts.format(texts)
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            # pred = net(image1.to(device),image2.to(device),image3.to(device),text1,text2,text3,text4)

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
        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ==== RMSE{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p,rmse))
        return np.mean(losses), rho_s, rho_p
    
