# Update: Feb 21st

**Context:**

We have Nvidia Jetson - ARM architecture (processor), Maxwell architecture (GPU, 2014) with 4GB RAM. CPU and GPU have shared memory. It has CUDA toolkit 10.2 and pytorch 1.1 with jetpack 4.6 (last SDK support from Nvidia for this platform for building AI solutions). We were exploring potential of these boards for training models as a means of using them in place of edge devices to decide which part of computation should be offloaded to cloud.

- **Where we left off?**
    - We were trying to train small models using GPU on the board for CIFAR10 dataset. From a set of models **(add set here)**, lightest one was selected to check the capabilities for training. After making sure that pytorch was indeed using GPU, we were getting out of memory errors during training.
    - Valgrind, htop, jtop, python memory profiler (**show valgrind output here**): were used to determine how much  memory was being used during training. Multiple tools to paint an accurate enough picture. Various combinations tried. (**tell if asked**)
        - Using GPU increased memory usage by alot and led to out of memory errors. Why? Specifically when data was moved to the GPU memory then we saw exponential growth in usage.
        - Now pytorch expects GPU and CPU to have their own respective memories not a shared memory architecture.
        - The board is intended for inference tasks and not full-fledged training. The increased memory usage was partly due to loading of CUDA libraries and loading CUDA context so some overhead of 700-800MB is expected. We saw how much? For inference only, using tensorRT is suggested to keep this overhead minimal. Specially made by Nvidia.
- **What was done?**
    - We want to know how the memory is used.
    Total 
    = CPU + GPU
    = regular memory usage + CUDA memory usage
    = (OS + steady state + desktop GUI) + (programs: Python, dataset loaded, tensors declared, profiler if any) + (CUDA libraries, Vars for compute, tensors moved from CPU
    - Direct CUDA memory allocation using unified memory for CPU+GPU: [here](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/), allocating using cuda: [here](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
    - Looked into unified memory: [here](https://on-demand.gputechconf.com/gtc/2018/presentation/s8430-everything-you-need-to-know-about-unified-memory.pdf) , A paper trying to make unified memory implementation for pytorch: [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9599280) - a pytorch framework does not exist for before Pascal architecture: [here](https://www.nvidia.com/en-us/technologies/)
    - Memory allocation on GPU - CUDA context
    
    how is cuda memory managed:? [https://stackoverflow.com/questions/8684770/how-is-cuda-memory-managed](https://stackoverflow.com/questions/8684770/how-is-cuda-memory-managed)
    [https://stackoverflow.com/questions/64068771/pytorch-what-happens-to-memory-when-moving-tensor-to-gpu](https://stackoverflow.com/questions/64068771/pytorch-what-happens-to-memory-when-moving-tensor-to-gpu)
    [https://stackoverflow.com/questions/63061779/pytorch-when-do-i-need-to-use-todevice-on-a-model-or-tensor](https://stackoverflow.com/questions/63061779/pytorch-when-do-i-need-to-use-todevice-on-a-model-or-tensor)
    
    [Pinned memory](https://pytorch.org/docs/1.10/data.html?highlight=torch%20utils%20data%20dataloader#torch.utils.data.DataLoader) - data loader: memory that the operating system prevents from being swapped out to disk, which makes it faster for the GPU to access because the data does not need to be copied to a staging area in GPU memory before being moved to its final destination in GPU memory.
    
    [tensor.to](http://tensor.to/)("cuda") is a separate operation that explicitly moves a tensor from CPU to GPU memory.
    
    https://stackoverflow.com/questions/65327247/load-pytorch-dataloader-into-gpu

    1. Initialising directly on GPU: Initializing a tensor directly on the GPU allocates memory only on the GPU, avoiding this double allocation.
    2. Cleanup - garbage collection for dataloader on CPU + per batch cleanup on GPU
    
    gc.collect
    **torch.cuda.empty_cache():** When you call torch.cuda.empty_cache(), PyTorch releases the cached (unused) memory blocks held by the allocator back to the GPU, making them available for other GPU applications. However, it does not affect the memory of tensors that are currently in use.
    In your loop, when you assign new data to inputs and targets with each iteration, the old data on the GPU is replaced. If there are no other references to the old data, PyTorch marks that memory as free. However, "free" in PyTorch's allocator means it's available for future allocations by PyTorch operations, not that it has been returned to the GPU's free memory pool.
    
- Conclusions:
    - Initialising tensor directly on GPU for big significant enough samples (200000 * 1000) (simple Linear regression model) - leads to reduction in memory usage: from 3.6GB (CPU init then move to GPU) to 3.2GB (GPU init)
    - Cleanup gc + **torch.cuda.empty_cache() - no change in the memory usage for the LeNet training.**
- next steps:
    - Dataloader directly initiliase on GPU (how??)
- Advice needed:
    - Swap memory
    - Interconnected mode: board + PC: shared, async edge devices - for inference and small training before federated learning (partial model - instead of say words - we send only embeddings)
- Advice given:
    - pytorch: existing framework - use it to run a model on the board: make the dataloader optimisation.
    - TinyML: what can we fit on the boards? using tinyML tools.
    - LLM inference on mobile phones: gammy on mobile phones
    - Connecting multiple boards for training will be too complex at this time. Focus on something you can do by May. Talk to Rishi and Rafael.
- General notes from meeting:
    - finetuning the model in a shared environment.
    - keep it inference limited
    There are multiple base models and there are inference SLOs, if for model 1 GPUs are busy and for model 2, they are free, then you will divert the request accordingly.
    If a query comes, it will be associated with one of the related models.
    Concept of distance between two models and then sending the queries accordingly.
    - diff on models
    Inference in a shared environment

### Logistics:
- Next week Anne-marie will not be there in the meeting.
- You figure out how to use TinyML tools on this board.
- Figure out how to run inference for a transformer on this board.