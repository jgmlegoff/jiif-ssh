import argparse
import os
import yaml

import xarray as xr

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from sklearn.metrics import mean_squared_error

import models
from utils import make_coord
from test import batched_predict

from tqdm import tqdm

def resize_fn(img, size):
    return T.Resize(size, T.InterpolationMode.BICUBIC)(img)

def predict(ssh, sst, norms) : 
    gt = T.ToTensor()(ssh)
    input_lr = T.ToTensor()(ssh)
    input_hr = T.ToTensor()(sst)


    if args.inputsize is None : 
        h_lr = math.floor(gt.shape[-2] / int(args.scale) + 1e-9)
        w_lr = math.floor(gt.shape[-1] / int(args.scale) + 1e-9)
        gt = gt[:, :round(h_lr * int(args.scale)), :round(w_lr * int(args.scale))] # assume round int
    else : 
        w_lr = int(args.inputsize)
        h_lr = w_lr
        w_hr = round(w_lr * int(args.scale))
        h_hr = w_hr
        #x0 = random.randint(0, img.shape[-2] - w_hr)
        #y0 = random.randint(0, img.shape[-1] - w_hr)
        #x0, y0 =0,0
        #crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        gt = resize_fn(gt, w_hr)
    input_ssh = resize_fn(input_lr, h_lr)
    input_sst = resize_fn(input_hr, h_hr)
    input_ssh = (input_ssh - norms['inp_ssh']['sub'][0]) / norms['inp_ssh']['div'][0]
    input_sst = (input_sst - norms['inp_sst']['sub'][0]) / norms['inp_sst']['div'][0]
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = gt.shape[-2],gt.shape[-1]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, input_ssh.cuda().unsqueeze(0), input_sst.cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred*norms['gt']['div'][0] + norms['gt']['sub'][0]).view(h, w, 1).permute(2, 0, 1).cpu()
    input_ssh = (input_ssh * norms['inp_ssh']['div'][0]) + norms['inp_ssh']['sub'][0]
    input_sst = (input_sst * norms['inp_sst']['div'][0]) + norms['inp_sst']['sub'][0]

    bicubic = resize_fn(input_ssh, (h,w))

    return(pred, bicubic, gt, input_ssh, input_sst)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--savefig',default=False)
    parser.add_argument('--rmse',default=False)
    parser.add_argument('--inputsize')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    timestep = 5

    configpath = os.path.dirname(args.model)
    with open(os.path.join(configpath,'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    norms = config['data_norm']
    print('Normalisation coeff loaded.')

    if args.input == 'natl-sst':
        input_path = '/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_03_06_09_12-2008.npz'
        npz_data = np.load(input_path)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
        ssh_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]
        sst_data = npz_data["FdataAllVar"][1,:,:,0:n_lat]

        pred, bicubic, ssh_hr, ssh_down, sst_hr = predict(ssh_data[timestep], sst_data[timestep], norms)

    elif args.input == 'natl-sst-train':
        input_path = '/data/jean.legoff/data/RESAC-SARGAS60/data/natl60_htuv_01102012_01102013.npz'
        npz_data = np.load(input_path)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
        ssh_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]
        sst_data = npz_data["FdataAllVar"][1,:,:,0:n_lat]

        pred, bicubic, ssh_hr, ssh_down, sst_hr = predict(ssh_data[timestep], sst_data[timestep], norms)

    else : 
        print("Please choose a valid input configuration : 'natl', 'mercator' and 'natl-sst'.")
        exit()

    ########################
    ### Plotting results ###
    ########################

    fig, axs = plt.subplots(1, 4,figsize=(18, 9))
    axs[0].imshow(sst_hr.permute(1, 2, 0), cmap=plt.cm.RdBu)    
    axs[1].imshow((ssh_down).permute(1, 2, 0))
    axs[2].imshow((ssh_hr).permute(1, 2, 0))
    axs[3].imshow(sst_hr.permute(1, 2, 0))
    plt.show()

    fig, axs = plt.subplots(2, 1,figsize=(18, 9))
    pos0 = axs[0].imshow(sst_hr.permute(1, 2, 0), cmap=plt.cm.RdBu)    
    plt.colorbar(pos0, ax=axs[0])
    pos1 = axs[1].imshow((ssh_hr).permute(1, 2, 0))
    plt.colorbar(pos1, ax=axs[1])
    axs[0].title.set_text('Sea Surface Height')
    axs[1].title.set_text('Sea Surface Temperature')
    plt.show()

    print(ssh_hr.shape, pred.shape)
    fig, axs = plt.subplots(2, 3,figsize=(18, 9))
    pos00 = axs[0,0].imshow(ssh_hr.permute(1, 2, 0),vmin = -0.5,vmax = 0.8)
    plt.colorbar(pos00, ax=axs[0,0])
    pos01 = axs[0,2].imshow(sst_hr.permute(1, 2, 0), cmap=plt.cm.RdBu)
    plt.colorbar(pos01, ax=axs[0,2])
    pos10 = axs[1,0].imshow(pred.permute(1, 2, 0),vmin = -0.5,vmax = 0.8)
    plt.colorbar(pos10, ax=axs[1,0])
    pos02 = axs[0,1].imshow((ssh_down).permute(1, 2, 0))
    plt.colorbar(pos02, ax=axs[0,1])

    axs[0,0].title.set_text('Ground Truth')
    axs[0,1].title.set_text('Input LR')
    axs[0,2].title.set_text('Input HR')
    axs[1,0].title.set_text('Prediction')
    axs[1,1].title.set_text('Error Map')

    pos11 = axs[1,1].imshow(pred.permute(1, 2, 0) - ssh_hr.permute(1, 2, 0), cmap=mpl.cm.bwr,vmin = -0.5,vmax = 0.5)
    plt.colorbar(pos11, ax=axs[1,1])

    plt.show()
    
    if args.savefig :
        fig, axs = plt.subplots(2, 2,figsize=(18, 18))
        pos0 = axs[0,0].imshow(ssh_hr.permute(1, 2, 0),vmin = -0.6,vmax = 0.8)
        plt.colorbar(pos0, ax=axs[0,0], location = 'left', fraction=0.05)
        pos1 = axs[0,1].imshow((ssh_down).permute(1, 2, 0))
        plt.colorbar(pos1, ax=axs[0,1])
        pos2 = axs[1,0].imshow(pred.permute(1, 2, 0))
        plt.colorbar(pos2, ax=axs[1,0])
        
        pos3 = axs[1,1].imshow(pred.permute(1, 2, 0) - ssh_hr.permute(1, 2, 0), vmin = -0.35, vmax = 0.35, cmap=mpl.cm.bwr)
        plt.colorbar(pos3, ax=axs[1,1], fraction=0.05)
        fontsize = 30
        axs[0,0].title.set_text('Ground Truth')
        axs[0,1].title.set_text('Input')
        axs[1,0].title.set_text('Prediction')
        axs[1,1].title.set_text('Error map')
        for i in axs:
            for k in i:
                k.title.set_size(30)
        fig.suptitle(f"Super-Resolution scale : {args.scale}, Input size : {args.inputsize}, Dataset : {args.input}, Config : {args.model.split('/')[-2]}")
        plt.savefig(f"results{args.model.split('/')[-2]}_{args.input}_i{args.inputsize}_x{args.scale}.png")

    rmsepred = torch.flatten(pred)
    rmsegt = torch.flatten(ssh_hr.view(ssh_hr.shape[-2],ssh_hr.shape[-1],1))

    print("Network prediction : ", math.sqrt(mean_squared_error(np.nan_to_num(rmsepred, nan=np.nanmean(rmsepred)),np.nan_to_num(rmsegt, nan=np.nanmean(rmsegt)))))
    print("Bicubic prediction : ", math.sqrt(mean_squared_error(torch.flatten((bicubic).view(bicubic.shape[-2],bicubic.shape[-1],1)), torch.flatten(ssh_hr.view(ssh_hr.shape[-2],ssh_hr.shape[-1],1)))))
    #T.ToPILImage()(pred).save(args.output)

    if args.rmse : 
        rmses = []
        bic_rmses = []
        ssh_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]
        sst_data = npz_data["FdataAllVar"][1,:,:,0:n_lat]
        print(ssh_data.shape)
        for (ssh, sst) in tqdm(zip(ssh_data, sst_data)) : 
            pred, bicubic, img_hr, img_down, sst_hr = predict(ssh, sst, norms)
            rmsepred = torch.flatten(pred)
            rmsegt = torch.flatten(img_hr.view(img_hr.shape[-2],img_hr.shape[-1],1))
            rmses.append(math.sqrt(mean_squared_error(np.nan_to_num(rmsepred, nan=np.nanmean(rmsepred)),np.nan_to_num(rmsegt, nan=np.nanmean(rmsegt)))))

            bic_rmses.append(math.sqrt(mean_squared_error(torch.flatten(bicubic.view(bicubic.shape[-2],bicubic.shape[-1],1)),np.nan_to_num(rmsegt, nan=np.nanmean(rmsegt)))))

        print('Model MRMSE : ', np.mean(rmses))
        print('Bicubic MRMSE : ', np.mean(bic_rmses))

    
        print(rmses.index(min(rmses)))