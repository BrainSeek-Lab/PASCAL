# PASCAL: Precise and Efficient ANN-SNN Conversion using Spike Accumulation and Adaptive Layerwise Activation

Code for paper "PASCAL: Precise and Efficient ANN-SNN Conversion using Spike Accumulation and Adaptive Layerwise Activation"

# Training 

Training can be done using an arbitrary value of $L$ per layer. To update this, refer the procedure below:
* **VGG-16**: In the class `VGG()`, change the array `self.list` in the file [VGG.py](models/VGG.py). 
* **ResNet-18 and ResNet-34**: Change the global parameter `mylist` in the file [ResNet.py](models/ResNet.py).

After updating the value of $L$, you can train by using the following command:

**For CIFAR-10 and CIFAR-100**
```
python3 main_train.py -data={cifar10/cifar100} -arch={vgg16/resnet18/resnet34} -epochs={num_epochs} --id={myid}
```

**For ImageNet**
```
python main.py -a {vgg16/resnet18/resnet34} --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]

```
Checkpoints will be saved for every epoch, so that training can be resumed from any arbitrary point. In order to train as per the algorithm given in the paper for Adaptive Layerwise activation, you need to run two training iterations. First, create the initial checkpoint after 200 epochs (for CIFAR-10 and CIFAR-100) or 80 epochs (for ImageNet), and then update the value of $L$ layerwise, as specified above. Then, training can be resumed using the `--resume_from_ckpt=1` argument for the remaining epochs.  

# Inference

Inference is done using the PASCAL framework. You can specify the command line argument `data` as in the training case, but in order for inference to be correct, you also need to specify an array `levels` in [main_test.py](main_test.py) whose size is dependent on the number of convolution, FCN and Batch Normalization layers in the model. Additionally, you need to update the respective layerwise lists in VGG-16 and ResNet-18/34. The procedure is outlined below: 

## VGG-16
* If an `IF()` layer is succeeded by a convolution, then `levels` should contain three elements corresponding to it. 
* Otherwise, `levels` should have one element corresponding to it.
* Modify the list `self.list` in the file [VGG16.py](models/VGG16.py), and update the value of $L$ per layer. 

## ResNet-18 and ResNet-34
 
* If an `IF()` layer is succeeded by a convolution, then `levels` should contain two elements corresponding to it. 
* Otherwise, `levels` should have one element corresponding to it.
* Modify the global list `mylist` in the file [ResNet.py](models/ResNet.py), and update value of $L$ per layer.
* Note that since ResNet-s have shortcut layers, the value of $L$ corresponding to the shortcut should be the value for the previous layer, not the value for the current one. As an example, 
```python
mylist = mylist = [4,2,4,1,4,2,2,1,4,2,2,1,4,1,1,1,4] # for ResNet-18. 
# The corresponding value of `levels` in main:

levels = [4,4]+[2,2]+[4,4]+[1,1]+[4,4]+[2,2,4,4]+[2,2]+[1,1]+[4,4]+[2,2,4,4]+[2,2]+[1,1]+[4,4]+[1,1,4,4]+[1,1]+[1,1]+[4] 

```

## Running Inference 
```
python3 main_test.py -data={cifar10/cifar100/imagenet} -arch={vgg16/resnet18/resnet34} --id={myid} -arch={vgg16/resnet18/resnet34}
```

# Obtaining the AL Metric

Use the script `get_stats.py` to get the value of the AL metric. The list `npy_arr_list` should contain the layerwise outputs after QCFS activation. The script will output the value of the AL metric for each layer. 

# Obtaining the value of $T_{eff}$ 

Use the script `get_Teff.py` to get the value of $T_{eff}$. Update the value of `Tarr` at [line 76](https://github.com/BrainSeek-Lab/PASCAL/blob/1705941c8ec96a2e3f4ce216ec7751b4d5d8986a/get_Teff.py#L76) to the values of $L$ for each layer. Also, update the variable corresponding to the model and dataset at [line 77](https://github.com/BrainSeek-Lab/PASCAL/blob/1705941c8ec96a2e3f4ce216ec7751b4d5d8986a/get_Teff.py#L77). 

# Acknowledgements

This repository draws from the repository for the paper "Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks".
