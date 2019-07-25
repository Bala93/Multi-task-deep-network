import torch
import os
from tqdm import tqdm
from torch import nn
import numpy as np
import torchvision
from torch.nn import functional as F
import time



def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):
 
            _,inputs,targets,_,_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = F.nll_loss(outputs[0], targets.squeeze(1))
            losses.append(loss.item())

        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start

def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=False):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _,inputs,targets,_,_ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy() 
            output_final  = np.argmax(output_mask,axis=1).astype(float)*85

            output_final = torch.from_numpy(output_final).unsqueeze(1)
            
            if train=='True':
               save_image(targets.float(), 'Target_train', val_batch_size)
               save_image(output_final, 'Prediction_train', val_batch_size)
            else:
               save_image(targets.float(), 'Target', val_batch_size)
               save_image(output_final, 'Prediction', val_batch_size)
               
            break
