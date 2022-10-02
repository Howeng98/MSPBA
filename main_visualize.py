import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from codes import mvtecad
from codes.utils import resize, makedirpath

import torch


parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='bottle')
args = parser.parse_args()


def save_heatmaps(obj, maps):
    from skimage.segmentation import mark_boundaries
    N = maps.shape[0]
    images = mvtecad.get_x(obj, mode='test')
    masks = mvtecad.get_mask(obj)

    for n in tqdm(range(N)):
        fig, axes = plt.subplots(ncols=2)
        fig.set_size_inches(6, 3)

        image = resize(images[n], (128, 128))
        mask = resize(masks[n], (128, 128))
        image = mark_boundaries(image, mask, color=(1, 0, 0), mode='thick')

        axes[0].imshow(image)
        axes[0].set_axis_off()

        axes[1].imshow(maps[n], vmax=maps[n].max(), cmap='Reds')
        axes[1].set_axis_off()

        plt.tight_layout()
        fpath = f'anomaly_maps/heatmaps/{obj}/n{n:03d}.png'
        makedirpath(fpath)
        plt.savefig(fpath)
        plt.close()

        
def save_thresMaps(obj, maps, img_score, threshold):
    from skimage.segmentation import mark_boundaries
    N = maps.shape[0]
    images = mvtecad.get_x(obj, mode='test')
    masks = mvtecad.get_mask(obj)
    gt = masks.astype(np.int32)
    gt[gt == 255] = 1
    
    print(f'threshold: {threshold:.4f}')

    for n in tqdm(range(N)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        test_pred = maps[n]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1

        axes[0].imshow(images[n])
        axes[0].set_title('Image', fontsize=30)
        axes[0].set_axis_off()
        
        axes[1].imshow(gt[n], cmap='gray')
        axes[1].set_title('Ground Truth', fontsize=30)
        axes[1].set_axis_off()
        
        axes[2].imshow(test_pred, cmap='gray')
        axes[2].set_title('Predicted Mask', fontsize=30)
        axes[2].set_axis_off()

        plt.tight_layout()
        fpath = f'anomaly_maps/thresMaps/{obj}/n{n:03d}.png'
        makedirpath(fpath)
        plt.savefig(fpath)
        plt.close()


#########################


def main():
    from codes.inspection import eval_encoder_NN_multiK
    from codes.networks import Encoder64, Encoder32, Encoder16, Vgg16

    obj = args.obj

    vgg_model = Vgg16().cuda()
    enc64 = Encoder64(vgg_model, 64, 64).cuda()
    enc32 = Encoder32(vgg_model, 64, 64).cuda()
    enc16 = Encoder16(vgg_model, 64, 64).cuda()
    enc64.load_state_dict(torch.load(f'ckpts/{obj}/encoder64.pkl'))
    enc32.load_state_dict(torch.load(f'ckpts/{obj}/encoder32.pkl'))
    enc16.load_state_dict(torch.load(f'ckpts/{obj}/encoder16.pkl'))
    enc64.eval()
    enc32.eval()
    enc16.eval()
    
    start_time = time.time()
    results = eval_encoder_NN_multiK(enc64, enc32, enc16, obj)
    finish_time = time.time()
    time_elapsed = round((finish_time-start_time)/60, 2)
    print(f'time elapsed: {time_elapsed:.2f} minutes')
    
    maps = results['maps_sum']
    img_score = results['img_score']
    threshold = results['threshold']
    
    save_heatmaps(obj, maps)
    save_thresMaps(obj, maps, img_score, threshold)
    

if __name__ == '__main__':
    main()

