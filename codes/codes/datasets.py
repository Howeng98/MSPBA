import math
import random
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import *



__all__ = ['SVDD_Dataset', 'PositionDataset', 'KmeansDataset']



pos_to_diff = {
    0: (1, 0),
    1: (math.sqrt(3)/2, 1/2),
    2: (1/2, math.sqrt(3)/2),
    3: (0, 1),
    4: ((-1)*(1/2), math.sqrt(3)/2),
    5: ((-1)*(math.sqrt(3)/2), 1/2),
    6: (-1, 0),
    7: ((-1)*(math.sqrt(3)/2), (-1)*(1/2)),
    8: ((-1)*(1/2), (-1)*(math.sqrt(3)/2)),
    9: (0, -1),
    10: (1/2, (-1)*(math.sqrt(3)/2)),
    11: (math.sqrt(3)/2, (-1)*1/2)
}


def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w


def generate_coords_position(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    pos = np.random.randint(12)

    with task('P2'):
        J = K // 4 # J=16 or J=8

        K3_4 = 3 * K // 4 # K3_4=48 or K3_4=24
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + int(round(h_diff, 0))
        w2 = w1 + int(round(w_diff, 0))

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2, pos


def generate_coords_svdd(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1
        
        pos = np.random.randint(12)

    with task('P2'):
        J = K // 16

        h_jit, w_jit = 0, 0

        while h_jit == 0 and w_jit == 0:
            h_jit = np.random.randint(-J, J + 1) # -4~4 if K=64, -2~2 if K=32
            w_jit = np.random.randint(-J, J + 1) # -4~4 if K=64, -2~2 if K=32

        h2 = h1 + h_jit
        w2 = w1 + w_jit

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)
        
    # get negative example
    with task('p3'):
        J = K // 4 # J=16 or J=8

        K3_4 = 3 * K // 4 # K3_4=48 or K3_4=24
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)
        if h_diff < 0: h_diff-=K3_4
        else: h_diff += K3_4
        if w_diff < 0: w_diff-=K3_4
        else: w_diff += K3_4
        
        h2 = h1 + int(round(h_diff, 0))
        w2 = w1 + int(round(w_diff, 0))

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p3 = (h2, w2)

    return p1, p2, p3



class SVDD_Dataset(Dataset):
    def __init__(self, memmap, K=64, repeat=1):
        super().__init__()
        self.arr = np.asarray(memmap)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        K = self.K
        n = idx % N

        p1, p2, p3 = generate_coords_svdd(256, 256, K)

        image = self.arr[n]

        patch1 = crop_image_CHW(image, p1, K)
        patch2 = crop_image_CHW(image, p2, K)
        patch3 = crop_image_CHW(image, p3, K)

        return patch1, patch2, patch3

    @staticmethod
    def infer(enc, batch):
        x1s, x2s, x3s, = batch
        h1s = enc(x1s)
        h2s = enc(x2s)
        h3s = enc(x3s)
        diff = h1s - h2s # tensor, shape=(64,64,1,1)
        l2 = diff.norm(dim=1)
        loss = l2.mean()
        
        # introduce negative examples and maximize positive examples
        loss_pos = -F.cosine_similarity(h1s, h2s)
        loss_neg = F.cosine_similarity(h1s, h3s)
        
        loss += (loss_pos + loss_neg).mean()

        return loss


class PositionDataset(Dataset):
    def __init__(self, x, K=64, repeat=1):
        super(PositionDataset, self).__init__()
        self.x = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.x.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.x.shape[0]
        K = self.K
        n = idx % N

        image = self.x[n]
        p1, p2, pos = generate_coords_position(256, 256, K)

        patch1 = crop_image_CHW(image, p1, K).copy()
        patch2 = crop_image_CHW(image, p2, K).copy()

        # perturb RGB
        rgbshift1 = np.random.normal(scale=0.02, size=(3, 1, 1))
        rgbshift2 = np.random.normal(scale=0.02, size=(3, 1, 1))

        patch1 += rgbshift1
        patch2 += rgbshift2

        # additive noise
        noise1 = np.random.normal(scale=0.02, size=(3, K, K))
        noise2 = np.random.normal(scale=0.02, size=(3, K, K))

        patch1 += noise1
        patch2 += noise2

        return patch1, patch2, pos


class KmeansDataset(Dataset):
    def __init__(self, x, K=64, repeat=1):
        super().__init__()
        self.arr = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        K = self.K
        n = idx % N

        p1 = generate_coords(256, 256, K)
        image = self.arr[n]
        patch1 = crop_image_CHW(image, p1, K)

        return patch1