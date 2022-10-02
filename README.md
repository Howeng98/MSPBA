# Multi-Scale Patch-Based Representation Learning for Image Anomaly Detection and Segmentation ([paper link](https://openaccess.thecvf.com/content/WACV2022/html/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.html), published in WACV 2022)

<p align="left">
    <img alt="ViewCount" src="https://views.whatilearened.today/views/github/howeng98/MSPBA.svg">
    <a href='https://github.com/howeng98/MSPBA'><img alt='GitHub Clones' src='https://img.shields.io/badge/dynamic/json?color=success&label=clones&query=count&url=https://gist.githubusercontent.com/Howeng98/cbb010a4bc31a1d14b7d49c1d836635d/raw/clone.json&logo=github'></a>
</p>


*Authors - Chin-Chia Tsai, Tsung-Hsuan Wu, Shang-Hong Lai*

![image](https://user-images.githubusercontent.com/10960400/190884447-d513415f-13d3-4a28-ad0d-89b828c3fa0a.png)

**Abstract** - *Unsupervised representation learning has been proven to be effective for the challenging anomaly detection and segmentation tasks. In this paper, we propose a multi-scale patch-based representation learning method to extract critical and representative information from normal images. By taking the relative feature similarity between patches of different local distances into account, we can achieve better representation learning. Moreover, we propose a refined way to improve the self-supervised learning strategy, thus allowing our model to learn better geometric relationship between neighboring patches. Through sliding patches of different scales all over an image, our model extracts representative features from each patch and compares them with those in the training set of normal images to detect the anomalous regions. Our experimental results on MVTec AD dataset and BTAD dataset demonstrate the proposed method achieves the state-of-the-art accuracy for both anomaly detection and segmentation.*
<br />

## Framework
The framework is inspired from [PatchSVDD](https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch).

![image](https://user-images.githubusercontent.com/10960400/190884512-93ff110c-29c0-4c0a-9df3-ae894bd396b0.png)
<br />

## Dataset
[MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)<br />
[BTAD dataset](https://github.com/pankajmishra000/VT-ADL#beantech-anomaly-detection-dataset---btad)
<br />

## Usage
#### Step 1. Change the DATASET_PATH variable.
Set the [DATASET_PATH](https://github.com/chinchia/Defect-Detection/blob/12520d5caa88b381dc90d4047ae0cd7f9dcec837/codes/codes/mvtecad.py#L9) to the root path of MVTec AD dataset.

#### Step 2. Training.
```
python main_train.py --obj=bottle --lr=1e-5 --lambda_value=1e-3 --D=64
```

```obj``` denotes the name of the class in MVTec AD dataset.<br />
```lambda_value``` denotes the value of 'lambda' in Eq. 5 of the paper.<br />
```D``` denotes the number of embedding dimension.<br />
```lr``` denotes the learning rate of Adam optimizer.<br />
```groups_64``` denotes the groups of the cluster for patch size 64.<br />
```groups_32``` denotes the groups of the cluster for patch size 32.<br />
```groups_16``` denotes the groups of the cluster for patch size 16.

#### Step 3. Testing and get anomaly maps.
```
python main_visualize.py --obj=bottle
```

```obj``` denotes the name of the class in MVTec AD dataset.
<br />

## Qualitative Results
![image](https://user-images.githubusercontent.com/10960400/190885216-4cc6da86-83cd-4464-a6d5-0c71ad7aefd9.png)
<br />

## Experiments Results
![image](https://user-images.githubusercontent.com/10960400/190885257-031e3402-3410-41ad-8f2a-dd38a78ad2b2.png)
<br />

## Citation
```
Chin-Chia Tsai, Tsung-Hsuan Wu, and Shang-Hong Lai, "Multi-Scale Patch-Based Representation Learning for Image Anomaly Detection and Segmentation," 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 3065-3073, doi: 10.1109/WACV51458.2022.00312.
```

**BibTex**

```
@InProceedings{Tsai_2022_WACV,
    author    = {Tsai, Chin-Chia and Wu, Tsung-Hsuan and Lai, Shang-Hong},
    title     = {Multi-Scale Patch-Based Representation Learning for Image Anomaly Detection and Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3992-4000}
}
```
<br />
