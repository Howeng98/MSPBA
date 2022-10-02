import numpy as np
import shutil
import os
import scipy.spatial
import torch
from torchvision import transforms

__all__ = ['k_center', 'search_NN']


def k_center(features, groups:int, device="cuda"):
    c = torch.zeros([groups, features.shape[1]], device=device)
    id_size = torch.zeros(groups, device=device)
    distance = torch.zeros([features.shape[0],groups], device=device)
    min_dist = torch.zeros(1, device=device)+9999
    best_id = torch.zeros(features.shape[0], device=device)
    best_c = c
    for big_epoch in range(20):
        f_id = torch.randint(groups, size=[features.shape[0]],dtype=torch.int32, device=device)
        for epoch in range(30):
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
        total_dist = torch.zeros(1, device=device)
        for k in range(groups):
            total_dist += torch.mean(torch.sqrt(distance[f_id==k,k]))
        if total_dist<min_dist:
            min_dist = total_dist
            best_id = f_id
            best_c = c
            
    for k in range(groups):
        id_size[k] = torch.sum(best_id==k)
    return best_id, best_c


def search_NN(test_emb, train_emb_flat, NN=1, method='kdt'):
    if method == 'ngt':
        return search_NN_ngt(test_emb, train_emb_flat, NN=NN)

    else:
        from sklearn.neighbors import KDTree
        kdt = KDTree(train_emb_flat)
        
        Ntest, I, J, D = test_emb.shape # shape=(N,13,13,64) or shape=(N,57,57,64)
        closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
        l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)
    
        for n in range(Ntest):
            for i in range(I):
                dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
                closest_inds[n, i, :, :] = inds[:, :]
                l2_maps[n, i, :, :] = dists[:, :]
    
        return l2_maps, closest_inds


def search_NN_ngt(test_emb, train_emb_flat, NN=1):
    import ngtpy

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)
    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat)

    for n in range(Ntest):
        for i in range(I):
            for j in range(J):
                query = test_emb[n, i, j, :]
                results = index.search(query, NN)
                inds = [result[0] for result in results]

                closest_inds[n, i, j, :] = inds
                vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds



