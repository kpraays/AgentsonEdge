## to do
imagenet
mobilebert

time taken for one mini batch for different batch sizes: 8,16,32
forward pass time and backward pass time
add timer inside in the code: hardware counter

change the plots:
plot time vs accuracy

## imagenet
An images dataset organized with regards to the WordNet hierarchy. There are one lakh phrases in WordNet and each phrase is illustrated by on average 1000 images. It is a huge dataset of size hundred and fifty gigabytes.

## MobileNet()
MobileNetV1 is a lightweight neural network architecture specifically designed for mobile and edge devices with limited computational resources.

An improvement over MobileNet, MobileNetV2 introduces inverted residuals and linear bottlenecks for better performance.

## MobileBert
MobileNetV1 and MobileBERT are not the same; they are two different architectures designed for different purposes.

MobileBERT, on the other hand, is a variant of BERT (Bidirectional Encoder Representations from Transformers), which is a powerful natural language processing (NLP) model introduced by Google in 2018.

MobileBERT is designed to be more computationally efficient and suitable for deployment on mobile and edge devices while maintaining the language understanding capabilities of BERT.

It achieves efficiency by reducing the model size and the number of parameters while preserving the pre-training process to capture contextual information from large text corpora.

## ResNet18
ResNet18 (Residual Network): ResNet is a type of CNN introduced by Microsoft Research. It is known for its use of residual learning, which helps mitigate the vanishing gradient problem. ResNet18 is a specific variant with 18 layers.

PreActResNet18 (Pre-activated Residual Network):An improvement over the original ResNet, PreActResNet introduces pre-activation blocks, which involve applying batch normalization and activation functions before convolution operations.

ResNeXt29_2x64d: ResNeXt is another variant of ResNet that emphasizes the importance of cardinality (the number of paths through a block). ResNeXt29_2x64d refers to a ResNeXt model with a specific configuration of 29 layers and 2 groups of 64 channels each.


##### Iteration vs Epoch vs Batch size
- Iteration: An iteration refers to one update of the model's parameters using a single batch of data. Think of it as one loop for a subset of data. How long that loop would be is decided by the batch size.
- Epoch: Going once through all the samples in the dataset
- Batch size: Number of samples you are considering in one iteration.
    - Smaller the batch size: more noise but frequent updates of model parameters.
    - Larger batch size: less noise, more memory


## High memory usage

- Interesting point: edge devices data loading is also a slow process apart from training due to limited memory.

#### Some calculations:
- Datasize is not that great: why?
- Cifar-10: 50000/60000 samples for training - 32*32 RGB images
  - Size_Per_Sample=(32×32×3)×4 bytes (image data)+4 bytes (label)
  - Memory_Usage = batch_size × Size_Per_Sample
    - 2MB - 128 batch size, 16MB 1024 batch size

- Keeping the desktop GUI: 800MB
- swap getting full - multiple runs
- Could not explain why? We get 1.1GB memory usage by the program. (1024 batch size -- does not respond)

#### My theory:
GPU: 128-core Maxwell
For 128 cuda cores to execute threads in parallel - likely more than calculated memory is needed.
Each Stream Multiprocessor is designed to handle a specific number of CUDA cores and other resources + scheduling.

#### Observation:
GPU does not have separate memory. Whatever it has needs to go through single memoy: CPU+GPU - 3.9GB(usable) - There is some swap memory too from SD card (1.9GB)

After 1 epoch failing regardless of the batch size. (8/128)
This is what the memory usage looked like. ![memory-usage-faile-after-epoch](memory-usage.PNG)

##### Why?
Pytorch dataloader:
MultiProcessingDataLoaderIter --> trying to start a new process --> cannot allocate memory
    class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    Terminating multiprocessing logic: (keep number of workers to 0 -- will not spawn new processes - only main thread will handle data loading.)

##### Solution: Don't use worker threads for dataloading.
torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0)

Then what happens?
    peak 3.6-1.6 = 2GB
    (550CPU + 560GPU): 1.1GB in the jtop measurements per process --> python

## forward-backward pass - timer
counter on jetson boards are different - no inbuilt clock(hardware) - can use kernel linux abstractions
kernel time counter: ktime_get

python time module: perf_counter() - high-resolution, platform-specific timer
[to be done](https://forums.developer.nvidia.com/t/does-jetson-nano-has-hardware-timer-that-could-accurate-to-microseconds/164749/5)...

## Points to be noted:
The Maxwell and Pascal architectures had 128 CUDA cores per stream multiprocessor (SM). The integer unit was trimmed in maxwell architecture, removing the dedicated multiplication unit. (Also, on edge devices, to maintain power efficieny they remove additional processing units.) [specs](https://developer.nvidia.com/embedded/jetson-nano)

VRAM holds the assets, textures, shadow maps, and all other data being processed via GPU. The reason why graphic cards store this data in VRAM is that it is much faster to access it from VRAM compared to DRAM, SSD, or HDD.

In Jetson Nano: GPU does not have separate memory. Whatever it has needs to go through single memoy: CPU+GPU - 3.9GB.

## more calculations and thoughts
Data is being moved from SD card to memory for training. Batch size is influencing how fast it is being loaded. Cleanup is not instantaneous from memory because the RAM usage grows.

32*32*3*4Bytes*50000 ~600MB minimum + Data labels + How each model's architecture is made (training would also need some memory - layers declaration)

ResNet very quickly overwhelms such that not able to run with 128 batch size.
With smaller batch sizes, convergence will be slow.

- Check which part of code is responsible for freeing up the memory -- trigger it early.
- Make available greater memory on the board - increase swap.
- Pick low powered devices/ mobile specific architectures.
  - Their capabilities are limited: power/ efficieny trade off.

## done
- accuracy vs time
- mini-batch vs time
    - 8,16,32,128
    - 1024 (not responding)
    - LeNet

- measure memory consumption along with it.
- small batch sizes 32 or less - don't lead to much training (within 10 epochs)
  
## to be done
forward backward pass
longer training
memory issues resolve - look into specialised training for low powered devices (to get optimised metrics)

