import os
import time
import logging
import argparse
import numpy as np
from functools import reduce

import torch
from torch.utils.data import DataLoader

from codes import mvtecad
from codes.datasets import *
from codes.networks import *
from codes.utils import *
from codes.nearest_neighbor import k_center
from codes.inspection import eval_encoder_NN_multiK



parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='bottle', type=str)
parser.add_argument('--lambda_value', default=1e-3, type=float)
parser.add_argument('--D', default=64, type=int)
parser.add_argument('--epochs', default=101, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--groups_64', default=50, type=int)
parser.add_argument('--groups_32', default=50, type=int)
parser.add_argument('--groups_16', default=50, type=int)

args = parser.parse_args()


device = 'cuda'
torch.backends.cudnn.benchmark = True


# logging AUROC results
newline = '\n'
LOG = f'./log_result/AUROC_{args.obj}.log'
logging.basicConfig(filename=LOG, filemode="w", level=logging.INFO)
logging.info(f' [class:{args.obj}, lambda:{args.lambda_value}, learning rate:{args.lr}, total training epochs:{args.epochs}, groups_64:{args.groups_64}, groups_32:{args.groups_32}, groups_16:{args.groups_16}]{newline}{newline}')


if not os.path.isdir(f'ckpts/{args.obj}'):
    os.mkdir(f'ckpts/{args.obj}')


def train():
    obj = args.obj
    D = args.D
    lr = args.lr
    groups_64 = args.groups_64
    groups_32 = args.groups_32
    groups_16 = args.groups_16
    
    best_dec = 0
    best_seg = 0
        
    with task('Networks'):
        vgg_model = Vgg16().cuda()
        enc64 = Encoder64(vgg_model, 64, D).cuda()
        enc32 = Encoder32(vgg_model, 64, D).cuda()
        enc16 = Encoder16(vgg_model, 64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()
        cls_16 = PositionClassifier(16, D).cuda()

        modules = [enc64, enc32, enc16, cls_64, cls_32, cls_16]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        rep = 100
        datasets = dict()
        
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        datasets[f'pos_16'] = PositionDataset(train_x, K=16, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)
        datasets[f'svdd_16'] = SVDD_Dataset(train_x, K=16, repeat=rep)
        
        datasets[f'kmeans_64'] = KmeansDataset(train_x, K=64, repeat=rep)
        datasets[f'kmeans_32'] = KmeansDataset(train_x, K=32, repeat=rep)
        datasets[f'kmeans_16'] = KmeansDataset(train_x, K=16, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)

    # initialization for k-means
    features_toepoch5_64 = torch.zeros(datasets[f'kmeans_64'].__len__(), 64, 4, device=device)
    features_toepoch5_32 = torch.zeros(datasets[f'kmeans_32'].__len__(), 64, 4, device=device)
    features_toepoch5_16 = torch.zeros(datasets[f'kmeans_32'].__len__(), 64, 4, device=device)
    
    print('Start training')
    print(f'class:{obj}, lambda:{args.lambda_value}, learning rate:{lr}, epochs:{args.epochs}')
        
    for i_epoch in range(args.epochs):
    
        print(f'epoch {i_epoch}')
        start_time = time.time()
        
        total_loss = 0
        
        if i_epoch != 0:
            for module in modules:
                module.train()
            
            # initialization for k-means
            toepoch5_n = 0
            n_samples = 0
            features_64 = torch.zeros(datasets[f'kmeans_64'].__len__(), 64, device=device) # shape=(image_number*100,64)
            features_32 = torch.zeros(datasets[f'kmeans_32'].__len__(), 64, device=device) # shape=(image_number*100,64)
            features_16 = torch.zeros(datasets[f'kmeans_16'].__len__(), 64, device=device)
            total_id_64 = torch.zeros(datasets[f'kmeans_64'].__len__(), dtype=torch.int32, device=device) # group number of the image
            total_id_32 = torch.zeros(datasets[f'kmeans_32'].__len__(), dtype=torch.int32, device=device) # group number of the image
            total_id_16 = torch.zeros(datasets[f'kmeans_16'].__len__(), dtype=torch.int32, device=device)
            
            # clustering for first time
            if i_epoch == 5:
                features_toepoch5_64 = features_toepoch5_64.view(features_toepoch5_64.shape[0]*4,64) # shape=(image_number*100*4,64)
                features_toepoch5_32 = features_toepoch5_32.view(features_toepoch5_32.shape[0]*4,64) # shape=(image_number*100*4,64)
                features_toepoch5_16 = features_toepoch5_16.view(features_toepoch5_16.shape[0]*4,64)
                _, best_c_64 = k_center(features_toepoch5_64, groups=groups_64) # shape=(groups,64)
                _, best_c_32 = k_center(features_toepoch5_32, groups=groups_32) # shape=(groups,64)
                _, best_c_16 = k_center(features_toepoch5_16, groups=groups_16)

            for d in loader:
                d = to_device(d, device, non_blocking=True)
                
                opt.zero_grad()
                
                # calculate k-means loss (N=image number)
                feature_64 = (enc64(d['kmeans_64']).data)[:,:,0,0] # shape=(N,64)
                feature_32 = (enc32(d['kmeans_32']).data)[:,:,0,0] # shape=(N,64)
                feature_16 = (enc16(d['kmeans_16']).data)[:,:,0,0]
                
                if i_epoch < 5:
                    features_toepoch5_64[toepoch5_n:(toepoch5_n+feature_64.shape[0]),:,i_epoch-1] = feature_64.data
                    features_toepoch5_32[toepoch5_n:(toepoch5_n+feature_32.shape[0]),:,i_epoch-1] = feature_32.data
                    features_toepoch5_16[toepoch5_n:(toepoch5_n+feature_16.shape[0]),:,i_epoch-1] = feature_16.data
                    toepoch5_n += feature_64.shape[0]
                    
                elif i_epoch > 5 and i_epoch % 5 == 0:
                    distance_64 = torch.sum(torch.pow((feature_64.expand(best_c_64.shape[0],feature_64.shape[0],feature_64.shape[1]).permute(1,0,2)-best_c_64.unsqueeze(0)),2), dim=2) # shape=(N,groups)
                    distance_32 = torch.sum(torch.pow((feature_32.expand(best_c_32.shape[0],feature_32.shape[0],feature_32.shape[1]).permute(1,0,2)-best_c_32.unsqueeze(0)),2), dim=2) # shape=(N,groups)
                    distance_16 = torch.sum(torch.pow((feature_16.expand(best_c_16.shape[0],feature_16.shape[0],feature_16.shape[1]).permute(1,0,2)-best_c_16.unsqueeze(0)),2), dim=2) # shape=(N,groups)
                    y_id_64 = torch.argsort(distance_64, dim=1)[:,0].type(torch.int32)
                    y_id_32 = torch.argsort(distance_32, dim=1)[:,0].type(torch.int32)
                    y_id_16 = torch.argsort(distance_16, dim=1)[:,0].type(torch.int32)
                    features_64[n_samples:(n_samples+feature_64.shape[0])] = feature_64.data
                    features_32[n_samples:(n_samples+feature_32.shape[0])] = feature_32.data
                    features_16[n_samples:(n_samples+feature_16.shape[0])] = feature_16.data
                    total_id_64[n_samples:(n_samples+feature_64.shape[0])] = y_id_64
                    total_id_32[n_samples:(n_samples+feature_32.shape[0])] = y_id_32
                    total_id_16[n_samples:(n_samples+feature_16.shape[0])] = y_id_16
                    
                    n_samples += feature_64.shape[0]
                    
                    loss_kmeans_64 = distance_64.min(dim=1).values.mean()
                    loss_kmeans_32 = distance_32.min(dim=1).values.mean()
                    loss_kmeans_16 = distance_16.min(dim=1).values.mean()
                
                # SVDD loss and SSL loss
                loss_pos_64 = PositionClassifier.infer(cls_64, enc64, d['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc32, d['pos_32'])
                loss_pos_16 = PositionClassifier.infer(cls_16, enc16, d['pos_16'])
                loss_svdd_64 = SVDD_Dataset.infer(enc64, d['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc32, d['svdd_32'])
                loss_svdd_16 = SVDD_Dataset.infer(enc16, d['svdd_16'])
                
                if i_epoch <= 10: loss = loss_pos_64 + loss_pos_32 + loss_pos_16 + args.lambda_value * (loss_svdd_64 + loss_svdd_32 + loss_svdd_16)
                else: loss = loss_kmeans_64 + loss_kmeans_32 + loss_kmeans_16 + loss_pos_64 + loss_pos_32 + loss_pos_16 + args.lambda_value * (loss_svdd_64 + loss_svdd_32 + loss_svdd_16)
                total_loss += loss.item()
                
                loss.backward()
                opt.step()

        print(f'loss: {total_loss:.4f}')
        
        if i_epoch != 0:
            logging.info(f'epoch {i_epoch}')
            
            if i_epoch % 5 == 0:
                _, best_c_64, id_size_64 = k_center_simple(features_64, total_id_64, best_c_64, groups_64, device)
                _, best_c_32, id_size_32 = k_center_simple(features_32, total_id_32, best_c_32, groups_32, device)
                _, best_c_16, id_size_16 = k_center_simple(features_16, total_id_16, best_c_16, groups_16, device)
            if i_epoch > 10:
                aurocs = eval_encoder_NN_multiK(enc64, enc32, enc16, obj)
                curr_dec, curr_seg = log_result(obj, aurocs)
                
                if best_dec < curr_dec:
                    best_dec = curr_dec
                    best_seg = curr_seg
                    torch.save(enc64.state_dict(), f'ckpts/{obj}/encoder64.pkl')
                    torch.save(enc32.state_dict(), f'ckpts/{obj}/encoder32.pkl')
                    torch.save(enc16.state_dict(), f'ckpts/{obj}/encoder16.pkl')
                    np.save(f'./ckpts/{obj}/best_c_64.npy', best_c_64.detach().cpu().numpy())
                    np.save(f'./ckpts/{obj}/best_c_32.npy', best_c_32.detach().cpu().numpy())
                    np.save(f'./ckpts/{obj}/best_c_16.npy', best_c_16.detach().cpu().numpy())
                    
                elif best_dec == curr_dec and best_seg < curr_seg:
                    best_seg = curr_seg
                    torch.save(enc64.state_dict(), f'ckpts/{obj}/encoder64.pkl')
                    torch.save(enc32.state_dict(), f'ckpts/{obj}/encoder32.pkl')
                    torch.save(enc16.state_dict(), f'ckpts/{obj}/encoder16.pkl')
                    np.save(f'./ckpts/{obj}/best_c_64.npy', best_c_64.detach().cpu().numpy())
                    np.save(f'./ckpts/{obj}/best_c_32.npy', best_c_32.detach().cpu().numpy())
                    np.save(f'./ckpts/{obj}/best_c_16.npy', best_c_16.detach().cpu().numpy())
                        
                print(f'best detection auroc: {best_dec:4.1f}, segmentation auroc: {best_seg:4.1f}')        
        
        finish_time = time.time()
        time_elapsed = round((finish_time-start_time)/60, 2)
        print(f'time elapsed: {time_elapsed:.2f} minutes')


def log_result(obj, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100
    
    det_16 = aurocs['det_16'] * 100
    seg_16 = aurocs['seg_16'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    logging.info(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |K16| Det: {det_16:4.1f} Seg: {seg_16:4.1f} |sum| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj}){newline}')
    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |K16| Det: {det_16:4.1f} Seg: {seg_16:4.1f} |sum| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')
    return det_sum, seg_sum
    

def k_center_simple(features, f_id, c, groups:int, device="cpu"):
    id_size = torch.zeros(groups, device=device)
    distance = torch.zeros([features.shape[0],groups], device=device)

    for epoch in range(40):
        for k in range(groups):
            id_size[k] = torch.sum(f_id==k)
            if id_size[k] != 0:
                c[k,:] = torch.mean(features[f_id==k,:],dim=0)
        for k in range(groups):
            distance[:,k] = torch.sum(torch.pow((features - c[k,:]),2),dim = 1)
            new_id = torch.argsort(distance,dim = 1)[:,0].type(torch.int32)
        if torch.sum(torch.abs(f_id-new_id))==0:
            break
        else:
            f_id=new_id
            
    for k in range(groups):
        id_size[k] = torch.sum(f_id==k)
        
    print('id_size=',id_size.type(torch.int32))
    print(np.where(id_size.cpu()!=0)[0].shape)
    
    return f_id, c, id_size
  
  
if __name__ == '__main__':
    train()
