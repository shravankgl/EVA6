
# Assignment 9

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:
     1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
     2. Layer1 -
         1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
         2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
         3. Add(X, R1)
     3. Layer 2 -
         1. Conv 3x3 [256k]
         2. MaxPooling2D
         3. BN
         4. ReLU
     4. Layer 3 -
         1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
         2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
         3. Add(X, R2)
     5. MaxPooling with Kernel Size 4
     6. FC Layer
     7. SoftMax
2. Uses One Cycle Policy such that:
     1. Total Epochs = 24
     2. Max at Epoch = 5
     3. LRMIN = FIND
     4. LRMAX = FIND
     5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Target Accuracy: 93% 



# Image Augmentation
```
   train_transform = A.Compose(
        [
            A.Sequential([A.CropAndPad(px=4, keep_size=False), A.RandomCrop(32,32)]),
            #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            #A.HorizontalFlip(p=0.3),
            A.CoarseDropout(max_holes = 4, max_height=8, max_width=8, min_holes = 1,
                            min_height=8, min_width=8,
                            fill_value=mean, mask_fill_value = tuple([x * 255.0 for x in mean])),
            #A.Rotate (limit=5, p=0.5),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )
```

# Model 

```
QuickNet(
  (prep_layer): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (layer1): LayerBlock(
    (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (mp): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (rl): ReLU()
    (resblock): ResBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (rl): ReLU()
    )
  )
  (layer2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
  )
  (layer3): LayerBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (mp): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (rl): ReLU()
    (resblock): ResBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (rl): ReLU()
    )
  )
  (mp): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (fc_layer): Linear(in_features=512, out_features=10, bias=False)
)
```

# Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
         ResBlock-14          [-1, 128, 16, 16]               0
       LayerBlock-15          [-1, 128, 16, 16]               0
           Conv2d-16          [-1, 256, 16, 16]         294,912
        MaxPool2d-17            [-1, 256, 8, 8]               0
      BatchNorm2d-18            [-1, 256, 8, 8]             512
             ReLU-19            [-1, 256, 8, 8]               0
           Conv2d-20            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-21            [-1, 512, 4, 4]               0
      BatchNorm2d-22            [-1, 512, 4, 4]           1,024
             ReLU-23            [-1, 512, 4, 4]               0
           Conv2d-24            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
           Conv2d-27            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-28            [-1, 512, 4, 4]           1,024
             ReLU-29            [-1, 512, 4, 4]               0
         ResBlock-30            [-1, 512, 4, 4]               0
       LayerBlock-31            [-1, 512, 4, 4]               0
        MaxPool2d-32            [-1, 512, 1, 1]               0
           Linear-33                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.07
Params size (MB): 25.07
Estimated Total Size (MB): 32.15
----------------------------------------------------------------

```

# Training Logs
```
  0%|          | 0/391 [00:00<?, ?it/s]Epoch 0
loss=0.9820512533187866 batch_id=390: 100%|██████████| 391/391 [00:28<00:00, 13.51it/s]
Train set: Average loss: 0.0114, Accuracy: 24207/50000 (48.41%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -4.1864, Accuracy: 6323/10000 (63.23%)

Epoch 1
loss=1.0140113830566406 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.18it/s]
Train set: Average loss: 0.0079, Accuracy: 32602/50000 (65.20%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -5.9395, Accuracy: 7279/10000 (72.79%)

Epoch 2
loss=0.6586244702339172 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.23it/s]
Train set: Average loss: 0.0064, Accuracy: 35880/50000 (71.76%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.7286, Accuracy: 7458/10000 (74.58%)

Epoch 3
loss=0.7917295694351196 batch_id=390: 100%|██████████| 391/391 [00:28<00:00, 13.96it/s]
Train set: Average loss: 0.0056, Accuracy: 37695/50000 (75.39%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.5686, Accuracy: 7522/10000 (75.22%)

Epoch 4
loss=0.4791623651981354 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.07it/s]
Train set: Average loss: 0.0046, Accuracy: 39675/50000 (79.35%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.9430, Accuracy: 7962/10000 (79.62%)

Epoch 5
loss=0.5481140613555908 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.15it/s]
Train set: Average loss: 0.0041, Accuracy: 40982/50000 (81.96%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.9198, Accuracy: 7423/10000 (74.23%)

Epoch 6
loss=0.48495930433273315 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.08it/s]
Train set: Average loss: 0.0038, Accuracy: 41646/50000 (83.29%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.9548, Accuracy: 8442/10000 (84.42%)

Epoch 7
loss=0.370624840259552 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.06it/s]
Train set: Average loss: 0.0036, Accuracy: 42222/50000 (84.44%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.0214, Accuracy: 8368/10000 (83.68%)

Epoch 8
loss=0.3393266797065735 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.07it/s]
Train set: Average loss: 0.0033, Accuracy: 42842/50000 (85.68%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.0122, Accuracy: 8435/10000 (84.35%)

Epoch 9
loss=0.4991990923881531 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.06it/s]
Train set: Average loss: 0.0032, Accuracy: 43039/50000 (86.08%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -6.9029, Accuracy: 8338/10000 (83.38%)

Epoch 10
loss=0.33473241329193115 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.12it/s]
Train set: Average loss: 0.0031, Accuracy: 43252/50000 (86.50%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.0053, Accuracy: 8466/10000 (84.66%)

Epoch 11
loss=0.4427691400051117 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.10it/s]
Train set: Average loss: 0.0030, Accuracy: 43540/50000 (87.08%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.1908, Accuracy: 8571/10000 (85.71%)

Epoch 12
loss=0.3146713376045227 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.09it/s]
Train set: Average loss: 0.0028, Accuracy: 43818/50000 (87.64%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.4634, Accuracy: 8598/10000 (85.98%)

Epoch 13
loss=0.4381236135959625 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.05it/s]
Train set: Average loss: 0.0028, Accuracy: 43999/50000 (88.00%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.5919, Accuracy: 8753/10000 (87.53%)

Epoch 14
loss=0.3526217043399811 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.07it/s]
Train set: Average loss: 0.0026, Accuracy: 44456/50000 (88.91%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -7.3527, Accuracy: 8689/10000 (86.89%)

Epoch 15
loss=0.41579899191856384 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.05it/s]
Train set: Average loss: 0.0024, Accuracy: 44794/50000 (89.59%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -8.7327, Accuracy: 8769/10000 (87.69%)

Epoch 16
loss=0.24312397837638855 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.07it/s]
Train set: Average loss: 0.0022, Accuracy: 45237/50000 (90.47%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -8.2271, Accuracy: 8868/10000 (88.68%)

Epoch 17
loss=0.2687041461467743 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.05it/s]
Train set: Average loss: 0.0020, Accuracy: 45812/50000 (91.62%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -8.3293, Accuracy: 8944/10000 (89.44%)

Epoch 18
loss=0.3915124237537384 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.06it/s]
Train set: Average loss: 0.0017, Accuracy: 46502/50000 (93.00%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -8.5539, Accuracy: 8963/10000 (89.63%)

Epoch 19
loss=0.20456281304359436 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.07it/s]
Train set: Average loss: 0.0013, Accuracy: 47321/50000 (94.64%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -9.0152, Accuracy: 9076/10000 (90.76%)

Epoch 20
loss=0.11199953407049179 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.11it/s]
Train set: Average loss: 0.0010, Accuracy: 48078/50000 (96.16%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -9.0460, Accuracy: 9174/10000 (91.74%)

Epoch 21
loss=0.10104706138372421 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.08it/s]
Train set: Average loss: 0.0008, Accuracy: 48611/50000 (97.22%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -9.2683, Accuracy: 9220/10000 (92.20%)

Epoch 22
loss=0.11533256620168686 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.10it/s]
Train set: Average loss: 0.0006, Accuracy: 48955/50000 (97.91%)


  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: -9.2064, Accuracy: 9271/10000 (92.71%)

Epoch 23
loss=0.06154952198266983 batch_id=390: 100%|██████████| 391/391 [00:27<00:00, 14.07it/s]
Train set: Average loss: 0.0005, Accuracy: 49120/50000 (98.24%)



Test set: Average loss: -9.2541, Accuracy: 9264/10000 (92.92%)

```

# Misclassified Images

![mislassified](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/misclassified.png)


# GradCam
 For each misclassified image grid is created as following matrix
|   |   |   |   |   |
|---|---|---|---|---|
| Original Image | layer1_gradcam_heatmap | layer1_gradcam_plus_plus_heatmap|  layer1_gradcam_mask | layer1_gradcam_plus_plus_mask|
| Original Image | layer2_gradcam_heatmap | layer2_gradcam_plus_plus_heatmap|  layer2_gradcam_mask | layer2_gradcam_plus_plus_mask|
| Original Image | layer3_gradcam_heatmap | layer3_gradcam_plus_plus_heatmap|  layer3_gradcam_mask | layer3_gradcam_plus_plus_mask|
| Original Image | layer4_gradcam_heatmap | layer4_gradcam_plus_plus_heatmap|  layer4_gradcam_mask | layer4_gradcam_plus_plus_mask|

![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/1.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/2.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/3.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/4.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/5.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/6.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/7.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/8.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/9.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/10.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/11.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/12.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/13.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/14.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/15.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/16.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/17.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/18.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/19.png)
![mislassified gradcam](https://github.com/shravankgl/EVA6/blob/main/Session9/assets/20.png)
