
## Group 
 Shravan
 
 Vijay
 
 Sumsum
 
 Naveen 



## 1. what is your code all about? 

Created 3 functions to help in creating a model

 1)  **buildConvLayer** to do conv->activation->normalization->dropout based on arguments

```  
    def buildConvLayer(in_channels, out_channels, kernel_size = 3, padding = 0, bias = False, activation = nn.ReLU ,normalization = None, group_count = 2, dropout = None):
        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding, bias=bias))
        conv_layer.append(activation())
        if normalization:
            if "BN" == normalization:
                conv_layer.append(nn.BatchNorm2d(out_channels))
            if "GN" == normalization:
                conv_layer.append(nn.GroupNorm(group_count,out_channels))
            if "LN" == normalization:
                conv_layer.append(nn.GroupNorm(1,out_channels))

        if dropout:
            conv_layer.append(nn.Dropout(dropout))

        return conv_layer
```
 2) **buildConvBlock** to create a sequential convolution block(Dropout can be added at all layers or last layer only)\
```        
    def buildConvBlock(in_channels, out_channels_list, kernel_size = 3, padding = 0, bias = False, activation = nn.ReLU ,normalization = None, group_count = 2, dropout = None, dropout_layers = 'last'):
        conv_block = []
        dropout_val = None
        if dropout and 'all' == dropout_layers:
            dropout_val = dropout
        for out_channels in out_channels_list:
            conv_block += buildConvLayer(in_channels, out_channels, kernel_size, padding, bias, activation, normalization, group_count, dropout_val)
            in_channels = out_channels
        if dropout and 'last' == dropout_layers:
            conv_block.append(nn.Dropout(dropout))
        return nn.Sequential(*conv_block)
```

 3) **buildTransBlock** to create transition block 
```
    def buildTransBlock(in_channels, out_channels):
        trans_block = []
        if in_channels != out_channels:
            trans_block.append(buildConvLayer(in_channels, out_channels, kernel_size=1))
        trans_block.append(nn.AvgPool2d(2, 2))
        return nn.Sequential(*trans_block)
 ```

## 2. how to perform the 3 covered normalization     (cannot use values from the excel sheet shared)? 

 
 ### **Different type of Normalization** 

***Image normalization\*** 

The pixel values in images must be scaled prior to providing the images as input to a deep learning neural network model during the training or evaluation of the model. Traditionally, the images would have to be scaled prior to the development of the model and stored in memory or on disk in the scaled format. An alternative approach is to scale the images using a preferred scaling technique just-in-time during the training or model evaluation process

In rescaling the pixel values from 0-255 range to 0-1 range. The range in 0-1 scaling is known as **Normalization****.** *standard scaling*- Subtracting the dataset mean serves to "center" the data. Additionally, divide by the standard deviation of that feature or pixel as well to normalize each feature value to a z-score.

The Pixel scaling technique consists of three main types, 

- Pixel Normalization–     Scales values of the pixels in 0-1 range.
- Pixel Centring–     Scales values of the pixels to have a 0 mean.
- Pixel Standardization–     Scales values of the pixels to have 0 mean and unit (1) variance.

Other scaling method is Min-max scaling

***Batch normalization\*** 

A typical neural network is trained using a collected set of input data called **batch**. In a neural network, batch normalization is achieved through a normalization step that fixes the means and variances of each layer's inputs (eg channels of all images – all red, all blue, all green for RGB image). It is a two-step process. First, the input is normalized, and later rescaling and offsetting is performed. Ideally, the normalization would be conducted over the entire training set, but to use this step jointly with [stochastic optimization](https://en.wikipedia.org/wiki/Stochastic_optimization) methods, it is impractical to use the global information. Thus, normalization is restrained to each mini-batch in the training process. Training deep neural networks with tens of layers is challenging as they can be sensitive to the initial random weights and configuration of the learning algorithm. One possible reason for this difficulty is the distribution of the inputs to layers deep in the network may change after each mini-batch when the weights are updated. This can cause the learning algorithm to forever chase a moving target. This change in the distribution of inputs to layers in the network is referred to the technical name “*internal covariate shift*. Internal covariate shift will adversely affect training speed because the later layers have to adapt to this shifted distribution. By stabilizing the distribution, batch normalization minimizes the internal covariate shift and speed up training.

Example: (Figure 1) We are training an image classification model, that classifies the images into Dog or Not Dog. We have the images of white dogs only, these images will have certain distribution as well. Using these images model will update its parameters. if we get a new set of images, consisting of non-white dogs. These new images will have a slightly different distribution from the previous images. Now the model will change its parameters according to these new images. Hence the distribution of the hidden activation will also change. This change in hidden activation is known as an internal covariate shift.

*Limitations of [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)*

- You need to maintain running means.
- Doesn’t work with small batch sizes; large NLP models are usually     trained with small batch sizes.
- ![img](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/Figure%201%20Batch%20Normalization%20.png)
       Need     to compute means and variances across devices in distributed training.

 

***Layer Normalization\*** 

 

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![img](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/Figure%202.png) |


Layer normalization is a simpler normalization method that works on a wider range of settings. Layer normalization transforms the inputs to have zero mean and unit variance across the features. *Note that batch normalization fixes the zero mean and unit variance for each element.* Layer normalization does it for each batch across all elements. [Layer Normalization](https://arxiv.org/abs/1607.06450) which normalizes the activations along the feature direction instead of mini-batch direction. This overcomes the cons of BN by removing the dependency on batches and makes it easier to apply for RNNs as well.



***Group normalization\*** 

Group normalization normalizes values of the same sample and the same group of channels together. Group Normalization is also applied along the feature direction but unlike LN, it divides the features into certain groups and normalizes each group separately. In practice, Group normalization performs better than layer normalization, and its parameter *num_groups* is tuned as a hyper parameter.

***Regularized regression\*** 

A predictive model has to be as simple as possible, but no simpler. There is an important relationship between the complexity of a model and its usefulness in a learning context because of the following reasons:

• Simpler models are usually more generic and are more widely applicable (are generalizable)

• Simpler models require fewer training samples for effective training than the more complex ones

 

Regularization is a process used to create an optimally complex model, i.e. a model which is as simple as possible while performing well on the training data. Through regularization, the algorithm designer tries to strike the delicate balance between keeping the model simple, yet not making it too naive to be of any use. The regression does not account for model complexity - it only tries to minimize the error (e.g. MSE), although if it may result in arbitrarily complex coefficients. On the other hand, in regularized regression, the objective function has two parts - the **error term** and the **regularization term**.

 

There is ridge a(L2) and Lasso (L1) regression 

In ridge regression, an additional term of "sum of the squares of the coefficients" is added to the cost function along with the error term. In case of lasso regression, a regularisation term of "sum of the absolute value of the coefficients" is added

 



|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![img](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/Figure%203.png) |

## 3. show all 3 calculations for  4 sample 2x2 images (image shown in the content has 3 images)



 ![Normalization-Batch-4](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/Normalization-Batch-4.JPG)

## 4. your findings for normalization techniques,

 a) Batch Normalization + L1 + L2 
Params: 4.7k
Best Train Accuracy: 99.42
Best Test Accuracy: 99.21
 

 b) Layer Normalization + L2 
Params: 4.7k
Best Train Accuracy: 99.54
Best Test Accuracy: 99.26
  

 c) Group Normalization + L1 
Params: 4.7k
Best Train Accuracy: 99.33
Best Test Accuracy: 99.23

Layer Norm gave better accuracy of the 3

## 5. add all your graphs
![Metrics](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/graph.png)
 



 

## 6. your 3 collection-of-misclassified-images 
 a) Batch Normalization + L1 + L2 
 
 
 
  ![Batch Normalization + L1 + L2 ](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/BN.png)

 b) Layer Normalization + L2 
 
 
 
 ![Layer Normalization + L2 ](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/LN.png)

 c) Group Normalization + L1 
 
 
 
 ![Group Normalization + L1](https://github.com/shravankgl/EVA6/blob/main/6_Normalization_and_Regularization/assets/GN.png)


 

 

 

 

 

 

 

 


