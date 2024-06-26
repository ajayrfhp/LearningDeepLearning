## Task
- Look into the forward and backward phase of Dropout. You can check some numpy implementation out there									
- Pay attention to extra scaling operation that takes place in the forward pass. Understand, why it is there.									
- Think about how backprop works for Dropout. Any data from the forward pass needs to be stored?									
- Think if it is a good idea to apply dropout after a convolution layer (typically dropout is applied after fc layers).[PyTorchDropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html)									
- Look into droplayer (a.k.a, StochasticDepth) and [dropblock](understand the implementation: (https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth); 
  [example usage](https://github.com/rwightman/pytorch-image-models/blob/e98c93264cde1657b188f974dc928b9d73303b18/timm/models/rexnet.py#L98-L101)									
- Look into how dropout can be used to derive some [prediction confidence](https://pgg1610.github.io/blog_fastpages/python/pytorch/machine-learning/2021/01/11/Simple_Dropout.html)


## Notes
- Read dropout chapter from [D2L](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
- Introduce noise to input to prevent overfitting.
- Noise is injected by randomly making some set of weight as 0 during input and normalizing other neurons so that mean remains unchanged.
- Dropout can be used in test time to evaluate uncertainity of predictions.
  - If slightly different sets of weights are returning similar answers, it can be argued that model is certain about its predictions
  - We can thus argue model will provide similar answers for changes in inputs
- Mechanics
   - Forward pass
     ```
     for i in dimensions:
       h'[i] = h[i] with probability p
       h'[i] = (h[i] / 1-p) with probability 1-p

     h' should be same shape as h

     A correct mask can be achieved the transformation
     h' = mask * h
     
     ```
   - [My implementation](https://colab.research.google.com/drive/1c4cmicvGP2HXhUnY3NIHCGp8un3JhST_?authuser=0#scrollTo=3WY7BY1C2PeT)
  - Backward pass
    - The mask has be to stored for reverse mode differentiation
     ```
     dh' / dh = mask
     de / dw = (de / dh') * (dh' / dh)
     gradient_wrt_inputs = gradient_wrt_outputs * mask 
     ```
    - [Someone else's implementation](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/dropout_layer)
