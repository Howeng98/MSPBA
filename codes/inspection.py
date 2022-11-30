import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import DataLoader

from codes import mvtecad
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores



__all__ = ['eval_encoder_NN_multiK']



def sliding_window(image, step_size, window_size):
    arr_size = (image.shape[0]//step_size) ** 2
    windows = np.zeros((arr_size, 1))
    count = 0
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            windows[count, :] = image[y:y+window_size[1], x:x+window_size[0]].mean()
            count += 1
    return windows.max(axis=0)


def assess_anomaly_maps(obj, anomaly_maps, map_type=None):
    auroc_seg, threshold = mvtecad.segmentation_auroc(obj, anomaly_maps)
    
    # sliding window: size (8 x 8) pixels
    if map_type == 'multi':
        anomaly_scores = np.array([])
        for n in range(anomaly_maps.shape[0]):
            anomaly_scores = np.concatenate((anomaly_scores, sliding_window(anomaly_maps[n,:,:], 4, (8,8))))
        auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
        print(f'new_score_calculation: {auroc_det:.3f}')
    
    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg, anomaly_scores, threshold


def infer(x, enc, K, S):
    x = NHWC2NCHW(x)
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()
            
            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    
    return embs


#########################


def eval_encoder_NN_multiK(enc64, enc32, enc16, obj):
    
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')
    
    embs64_tr = infer(x_tr, enc64, K=64, S=16)
    embs64_te = infer(x_te, enc64, K=64, S=16)

    embs32_tr = infer(x_tr, enc32, K=32, S=4)
    embs32_te = infer(x_te, enc32, K=32, S=4)
    
    embs16_tr = infer(x_tr, enc16, K=16, S=4)
    embs16_te = infer(x_te, enc16, K=16, S=4)

    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te
    embs16 = embs16_tr, embs16_te
    
    return eval_embeddings_NN_multiK(obj, embs64, embs32, embs16)


def eval_embeddings_NN_multiK(obj, embs64, embs32, embs16, NN=1):
    
    emb_tr64, emb_te64 = embs64
    maps_64 = measure_emb_NN(emb_te64, emb_tr64, method='kdt', NN=NN)
    maps_64_distribute = distribute_scores(maps_64, (256, 256), K=64, S=16)
    det_64, seg_64, _, _ = assess_anomaly_maps(obj, maps_64_distribute)

    emb_tr32, emb_te32 = embs32
    maps_32 = measure_emb_NN(emb_te32, emb_tr32, method='ngt', NN=NN)
    maps_32_distribute = distribute_scores(maps_32, (256, 256), K=32, S=4)
    det_32, seg_32, _, _ = assess_anomaly_maps(obj, maps_32_distribute)
    
    emb_tr16, emb_te16 = embs16
    maps_16 = measure_emb_NN(emb_te16, emb_tr16, method='ngt', NN=NN)
    maps_16_distribute = distribute_scores(maps_16, (256, 256), K=16, S=4)
    det_16, seg_16, _, _ = assess_anomaly_maps(obj, maps_16_distribute)

    maps_sum = maps_64_distribute + maps_32_distribute + maps_16_distribute
    det_sum, seg_sum, imgs_score, threshold = assess_anomaly_maps(obj, maps_sum, map_type='multi')

    maps_mult = maps_64_distribute * maps_32_distribute * maps_16_distribute
    det_mult, seg_mult, _, _ = assess_anomaly_maps(obj, maps_mult, map_type='multi')

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,
        
        'det_16': det_16,
        'seg_16': seg_16,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64_distribute,
        'maps_32': maps_32_distribute,
        'maps_16': maps_16_distribute,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
        
        'img_score': imgs_score,
        'threshold': threshold
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
    