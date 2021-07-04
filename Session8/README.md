# Assignment 8

1. Train Resnet18 for 40 Epochs
2. 20 misclassified images
3. 20 GradCam output on the SAME misclassified images
4. Apply these transforms while training:
  1.RandomCrop(32, padding=4)
  2.CutOut(16x16)
  3.Rotate(±5°)
5. Must use ReduceLROnPlateau
6. Must use LayerNormalization ONLY


# Image Augmentation
```
train_transform = A.Compose(
        [
            A.Sequential([A.CropAndPad(px=4, keep_size=False), A.RandomCrop(32,32)]),
            #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            #A.HorizontalFlip(p=0.3),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1,
                            min_height=16, min_width=16,
                            fill_value=mean, mask_fill_value = None),
            A.Rotate (limit=5, p=0.5),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )
```

# Model 

```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): GroupNorm(1, 64, eps=1e-05, affine=True)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 64, eps=1e-05, affine=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 64, eps=1e-05, affine=True)
      (shortcut): Sequential()
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 64, eps=1e-05, affine=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 64, eps=1e-05, affine=True)
      (shortcut): Sequential()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 128, eps=1e-05, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 128, eps=1e-05, affine=True)
      (shortcut): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): GroupNorm(1, 128, eps=1e-05, affine=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 128, eps=1e-05, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 128, eps=1e-05, affine=True)
      (shortcut): Sequential()
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 256, eps=1e-05, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 256, eps=1e-05, affine=True)
      (shortcut): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): GroupNorm(1, 256, eps=1e-05, affine=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 256, eps=1e-05, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 256, eps=1e-05, affine=True)
      (shortcut): Sequential()
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 512, eps=1e-05, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 512, eps=1e-05, affine=True)
      (shortcut): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): GroupNorm(1, 512, eps=1e-05, affine=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): GroupNorm(1, 512, eps=1e-05, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): GroupNorm(1, 512, eps=1e-05, affine=True)
      (shortcut): Sequential()
    )
  )
  (linear): Linear(in_features=512, out_features=10, bias=True)
)
```

# Training Logs
```
  0%|          | 0/391 [00:00<?, ?it/s]

Epoch 0

loss=2.1109790802001953 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.62it/s]


Train set: Average loss: 0.0177, Accuracy: 8385/50000 (16.77%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -1.6312, Accuracy: 2780/10000 (27.80%)

Epoch 1

loss=1.8608968257904053 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.66it/s]


Train set: Average loss: 0.0149, Accuracy: 14809/50000 (29.62%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -1.7134, Accuracy: 3744/10000 (37.44%)

Epoch 2

loss=1.6681989431381226 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.64it/s]


Train set: Average loss: 0.0138, Accuracy: 18012/50000 (36.02%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -2.0326, Accuracy: 4095/10000 (40.95%)

Epoch 3

loss=1.4567021131515503 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.62it/s]


Train set: Average loss: 0.0127, Accuracy: 20662/50000 (41.32%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -2.4295, Accuracy: 4480/10000 (44.80%)

Epoch 4

loss=1.3668668270111084 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.63it/s]


Train set: Average loss: 0.0117, Accuracy: 23082/50000 (46.16%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -2.6634, Accuracy: 5206/10000 (52.06%)

Epoch 5

loss=1.4380391836166382 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.64it/s]


Train set: Average loss: 0.0109, Accuracy: 24899/50000 (49.80%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.0377, Accuracy: 5442/10000 (54.42%)

Epoch 6

loss=1.3053791522979736 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.64it/s]


Train set: Average loss: 0.0102, Accuracy: 26632/50000 (53.26%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.1573, Accuracy: 5586/10000 (55.86%)

Epoch 7

loss=1.1441553831100464 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.65it/s]


Train set: Average loss: 0.0095, Accuracy: 28103/50000 (56.21%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.4908, Accuracy: 5845/10000 (58.45%)

Epoch 8

loss=1.0378413200378418 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.66it/s]


Train set: Average loss: 0.0089, Accuracy: 29557/50000 (59.11%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.5339, Accuracy: 6162/10000 (61.62%)

Epoch 9

loss=1.0398728847503662 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.65it/s]


Train set: Average loss: 0.0084, Accuracy: 30829/50000 (61.66%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -3.6100, Accuracy: 6070/10000 (60.70%)

Epoch 10

loss=0.8645939826965332 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.65it/s]


Train set: Average loss: 0.0078, Accuracy: 32163/50000 (64.33%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.3410, Accuracy: 6536/10000 (65.36%)

Epoch 11

loss=1.091718316078186 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.64it/s]


Train set: Average loss: 0.0074, Accuracy: 33177/50000 (66.35%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.3571, Accuracy: 6542/10000 (65.42%)

Epoch 12

loss=0.7277988791465759 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.65it/s]


Train set: Average loss: 0.0070, Accuracy: 34230/50000 (68.46%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.6634, Accuracy: 6842/10000 (68.42%)

Epoch 13

loss=0.8883625268936157 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.67it/s]


Train set: Average loss: 0.0066, Accuracy: 35108/50000 (70.22%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -4.3702, Accuracy: 6974/10000 (69.74%)

Epoch 14

loss=0.6497334837913513 batch_id=390: 100%|██████████| 391/391 [01:09<00:00,  5.66it/s]


Train set: Average loss: 0.0062, Accuracy: 36100/50000 (72.20%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.1173, Accuracy: 7243/10000 (72.43%)

Epoch 15

loss=0.8718096613883972 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0058, Accuracy: 36823/50000 (73.65%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.2945, Accuracy: 7401/10000 (74.01%)

Epoch 16

loss=0.6682058572769165 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.67it/s]


Train set: Average loss: 0.0055, Accuracy: 37567/50000 (75.13%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.1577, Accuracy: 7272/10000 (72.72%)

Epoch 17

loss=0.5900375247001648 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.67it/s]


Train set: Average loss: 0.0052, Accuracy: 38131/50000 (76.26%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.5117, Accuracy: 7558/10000 (75.58%)

Epoch 18

loss=0.4641098380088806 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0049, Accuracy: 38941/50000 (77.88%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.5521, Accuracy: 7677/10000 (76.77%)

Epoch 19

loss=0.551015317440033 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0047, Accuracy: 39391/50000 (78.78%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3773, Accuracy: 7654/10000 (76.54%)

Epoch 20

loss=0.5431657433509827 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.67it/s]


Train set: Average loss: 0.0045, Accuracy: 39906/50000 (79.81%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -5.9380, Accuracy: 7683/10000 (76.83%)

Epoch 21

loss=0.5130332708358765 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0043, Accuracy: 40352/50000 (80.70%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.2641, Accuracy: 7889/10000 (78.89%)

Epoch 22

loss=0.6191381216049194 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0041, Accuracy: 40760/50000 (81.52%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.3619, Accuracy: 7944/10000 (79.44%)

Epoch 23

loss=0.3500906825065613 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0039, Accuracy: 41217/50000 (82.43%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.7286, Accuracy: 7928/10000 (79.28%)

Epoch 24

loss=0.469266414642334 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.67it/s]


Train set: Average loss: 0.0037, Accuracy: 41587/50000 (83.17%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.2149, Accuracy: 8019/10000 (80.19%)

Epoch 25

loss=0.33035674691200256 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0035, Accuracy: 42008/50000 (84.02%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.0745, Accuracy: 8022/10000 (80.22%)

Epoch 26

loss=0.648070752620697 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0034, Accuracy: 42340/50000 (84.68%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -6.8767, Accuracy: 7766/10000 (77.66%)

Epoch 27

loss=0.42628470063209534 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0033, Accuracy: 42597/50000 (85.19%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.1939, Accuracy: 8018/10000 (80.18%)

Epoch 28

loss=0.4752834737300873 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0031, Accuracy: 43113/50000 (86.23%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.0190, Accuracy: 8160/10000 (81.60%)

Epoch 29

loss=0.5045437216758728 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0030, Accuracy: 43387/50000 (86.77%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.4041, Accuracy: 8099/10000 (80.99%)

Epoch 30

loss=0.5186904072761536 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0029, Accuracy: 43553/50000 (87.11%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.7485, Accuracy: 8099/10000 (80.99%)

Epoch 31

loss=0.2490675151348114 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.67it/s]


Train set: Average loss: 0.0028, Accuracy: 43703/50000 (87.41%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.7923, Accuracy: 8109/10000 (81.09%)

Epoch 32

loss=0.299244225025177 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0027, Accuracy: 43915/50000 (87.83%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -8.0115, Accuracy: 8226/10000 (82.26%)

Epoch 33

loss=0.3222663998603821 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0025, Accuracy: 44427/50000 (88.85%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -8.0999, Accuracy: 8105/10000 (81.05%)

Epoch 34

loss=0.3348537087440491 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0024, Accuracy: 44434/50000 (88.87%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -7.5515, Accuracy: 8176/10000 (81.76%)

Epoch 35

loss=0.20978382229804993 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0023, Accuracy: 44677/50000 (89.35%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -8.4833, Accuracy: 8075/10000 (80.75%)

Epoch 36

loss=0.3616984188556671 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0023, Accuracy: 44875/50000 (89.75%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -8.4618, Accuracy: 8108/10000 (81.08%)

Epoch 37

loss=0.29722294211387634 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]


Train set: Average loss: 0.0022, Accuracy: 44882/50000 (89.76%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -8.2261, Accuracy: 8214/10000 (82.14%)

Epoch 38

loss=0.22909784317016602 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.70it/s]


Train set: Average loss: 0.0021, Accuracy: 45140/50000 (90.28%)


  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: -9.5967, Accuracy: 8259/10000 (82.59%)

Epoch 39

loss=0.3203888535499573 batch_id=390: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]


Train set: Average loss: 0.0021, Accuracy: 45394/50000 (90.79%)



Test set: Average loss: -8.3232, Accuracy: 8227/10000 (82.27%)

```

# Misclassified Images

![mislassified](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/misclassified.png)


# GradCam
 For each misclassified image grid is created as following matrix
|   |   |   |   |   |
|---|---|---|---|---|
| Original Image | layer1_gradcam_heatmap | layer1_gradcam_plus_plus_heatmap|  layer1_gradcam_mask | layer1_gradcam_plus_plus_mask|
| Original Image | layer2_gradcam_heatmap | layer2_gradcam_plus_plus_heatmap|  layer2_gradcam_mask | layer2_gradcam_plus_plus_mask|
| Original Image | layer3_gradcam_heatmap | layer3_gradcam_plus_plus_heatmap|  layer3_gradcam_mask | layer3_gradcam_plus_plus_mask|
| Original Image | layer4_gradcam_heatmap | layer4_gradcam_plus_plus_heatmap|  layer4_gradcam_mask | layer4_gradcam_plus_plus_mask|

![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/1.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/2.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/3.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/4.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/5.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/6.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/7.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/8.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/9.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/10.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/11.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/12.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/13.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/14.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/15.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/16.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/17.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/18.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/19.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session8/assets/20.png)
