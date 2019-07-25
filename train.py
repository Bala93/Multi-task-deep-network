import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import numpy as np
import h5py
import torchvision
import random
from tensorboardX import SummaryWriter

from utils import visualize,evaluate
from losses import LossMulti
from models import UNet,UNet_DCAN,UNet_DMTN,PsiNet,UNet_ConvMCD
from dataset import DatasetImageMaskContourDist


class LossUNet:

    def __init__(self,weights=[1,1,1]):
    
        self.criterion = LossMulti(num_classes=2)
   
    def __call__(self,outputs,targets):
 
        criterion = self.criterion(outputs,targets)

        return criterion

class LossDCAN:

    def __init__(self,weights=[1,1,1]):
    
        self.criterion1 = LossMulti(num_classes=2)
        self.criterion2 = LossMulti(num_classes=2)
        self.weights = weights
   
    def __call__(self,outputs1,outputs2,targets1,targets2):
       
        criterion = self.weights[0] * self.criterion1(outputs1,targets1) + self.weights[1] * self.criterion2(outputs2,targets2)

        return criterion

class LossDMTN:

    def __init__(self,weights=[1,1,1]):
    
        self.criterion1 = LossMulti(num_classes=2)
        self.criterion2 = nn.MSELoss()
        self.weights = weights
   
    def __call__(self,outputs1,outputs2,targets1,targets2):

        criterion = self.weights[0] * self.criterion1(outputs1,targets1) + self.weights[1] * self.criterion2(outputs2,targets2)

        return criterion

class LossPsiNet:

    def __init__(self,weights=[1,1,1]):

        self.criterion1 = LossMulti(num_classes=2)
        self.criterion2 = LossMulti(num_classes=2)
        self.criterion3 = nn.MSELoss()
        self.weights = weights 
   
    def __call__(self,outputs1,outputs2,outputs3,targets1,targets2,targets3):

        criterion = self.weights[0] * self.criterion1(outputs1,targets1) + self.weights[1] * self.criterion2(outputs2,targets2) + self.weights[2] * self.criterion3(outputs3,targets3)

        return criterion
 
def define_loss(loss_type,weights=[1,1,1]):

    if loss_type == 'unet':
        criterion = LossUNet(weights)
    if loss_type == 'dcan':
        criterion = LossDCAN(weights)
    if loss_type == 'dmtn':
        criterion = LossDMTN(weights)
    if loss_type == 'psinet' or loss_type == 'convmcd':
        # Both psinet and convmcd uses same mask,contour and distance loss function
        criterion = LossPsiNet(weights)

    return criterion


def build_model(model_type):

    if model_type == 'unet':
        model = UNet(num_classes=2)
    if model_type == 'dcan':
        model = UNet_DCAN(num_classes=2)
    if model_type == 'dmtn':
        model = UNet_DMTN(num_classes=2)
    if model_type == 'psinet':
        model = PsiNet(num_classes=2)
    if model_type == 'convmcd':
        model = UNet_ConvMCD(num_classes=2)
  
    return model 

def train_model(model,targets,model_type,criterion,optimizer):

    if model_type == 'unet':

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs) 
            loss = criterion(outputs[0],targets[0])
            loss.backward()
            optimizer.step()

    if model_type == 'dcan':

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs) 
            loss = criterion(outputs[0],outputs[1],targets[0],targets[1])
            loss.backward()
            optimizer.step()

    if model_type == 'dmtn':

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs) 
            loss = criterion(outputs[0],outputs[1],targets[0],targets[2])
            loss.backward()
            optimizer.step()

    if model_type == 'psinet' or model_type == 'convmcd':

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs) 
            loss = criterion(outputs[0],outputs[1],outputs[2],targets[0],targets[1],targets[2])
            loss.backward()
            optimizer.step()

    return loss


if __name__ == "__main__":

    train_path  = '/media/htic/NewVolume3/Balamurali/polyp-segmentation/train_valid/train/image/*.jpg'
    val_path  = '/media/htic/NewVolume3/Balamurali/polyp-segmentation/train_valid/test/image/*.jpg'
    object_type = 'polyp'#polyp
    model_type = 'convmcd'
    distance_type = 'dist_mask' #dist_contour,dist_signed
    save_path = '/media/htic/NewVolume5/midl_experiments/nll/{}_{}/models_global'.format(object_type,model_type)

    use_pretrained = False
    pretrained_model_path = '/media/htic/NewVolume5/midl_experiments/nll/prostate_unet/models_run3/40.pt'
    #TODO:Add hyperparams to ArgParse. 
    batch_size = 16
    val_batch_size = 9 
    no_of_epochs = 150

    cuda_no = 0
    CUDA_SELECT = "cuda:{}".format(cuda_no)

    #TODO:Change the summary writer snippet 
    log_path = '/media/htic/NewVolume5/midl_experiments/nll/{}_{}/models_global/summary'.format(object_type,model_type)
    writer = SummaryWriter(log_dir=log_path)

    logging.basicConfig(filename="log_{}_run_global.txt".format(object_type),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M',
                            level=logging.INFO)
    logging.info('Model: UNet + Loss: FocalLoss(alpha=4) {}'.format(object_type)) 

    train_file_names = glob.glob(train_path)
    random.shuffle(train_file_names)
    val_file_names = glob.glob(val_path)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    model = build_model(model_type)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

    model = model.to(device)

    # To handle epoch start number and pretrained weight 
    epoch_start = '0'
    if(use_pretrained):
        print("Loading Model {}".format(os.path.basename(pretrained_model_path)))
        model.load_state_dict(torch.load(pretrained_model_path))
        epoch_start = os.path.basename(pretrained_model_path).split('.')[0]
        print(epoch_start)

    
    trainLoader   = DataLoader(DatasetImageMaskContourDist(train_file_names,distance_type),batch_size=batch_size)
    devLoader     = DataLoader(DatasetImageMaskContourDist(val_file_names,distance_type))
    displayLoader = DataLoader(DatasetImageMaskContourDist(val_file_names,distance_type),batch_size=val_batch_size)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = define_loss(model_type)

    for epoch in tqdm(range(int(epoch_start)+1,int(epoch_start)+1+no_of_epochs)):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i,(img_file_name,inputs,targets1,targets2,targets3) in enumerate(tqdm(trainLoader)):

            model.train()

            inputs    = inputs.to(device)
            targets1  = targets1.to(device)
            targets2  = targets2.to(device)
            targets3  = targets3.to(device)
 
            targets   = [targets1,targets2,targets3]

            loss = train_model(model,targets,model_type,criterion,optimizer)

            writer.add_scalar('loss', loss.item(), epoch)

            running_loss += loss.item()*inputs.size(0)

        epoch_loss = running_loss / len(train_file_names)

        if epoch%1 == 0:

            dev_loss,dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar('loss_valid', dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss,dev_loss))
        else:
            print("Global Loss:{} ".format(epoch_loss))
        
        logging.info('epoch:{} train_loss:{} '.format(epoch,epoch_loss))
        #if epoch%5 == 0:
        #    torch.save(model.state_dict(),os.path.join(save_path,str(epoch)+'.pt'))
