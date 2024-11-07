Notes:

- Function to be run on gpu is called kernel for CUDA.
- Code that runs on gpu is called device code.
- Code that runs on cpu is called the host code.

Nvidia jetson board has Cuda 10.2
Add the nvcc to path.
Terminal executes this file before starting: nano /home/username/.bashrc
- Export nvcc to path in this file.
export PATH="/usr/local/cuda-10.2/bin:$PATH"`
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"

dont keep threads in htop using shift+h



Data is loaded from datasets like CIFAR10 using PyTorch's DataLoader since the data loaded this way is inherently on the CPU and requires a transfer step to the GPU.
data loader to load data into pinned (or page-locked) memory. Pinned memory is a special area of CPU memory that the operating system prevents from being swapped out to disk, which makes it faster for the GPU to access because the data does not need to be copied to a staging area in GPU memory before being moved to its final destination in GPU memory.

Using tensor.to("cuda") is a separate operation that explicitly moves a tensor from CPU to GPU memory.

Initializing a tensor directly on the GPU allocates memory only on the GPU, avoiding this double allocation.

Direct GPU Tensor Initialization for Data and Targets: Modify your data loading process to ensure that the inputs and targets are directly placed on the GPU at the time they are loaded from the dataset, rather than moving them to the GPU in the training loop. This can be achieved by customizing the collate_fn of your DataLoader to move batches to the GPU as they are created

cleanup: 
gc.collect
torch.cuda.empty_cache(): When you call torch.cuda.empty_cache(), PyTorch releases the cached (unused) memory blocks held by the allocator back to the GPU, making them available for other GPU applications. However, it does not affect the memory of tensors that are currently in use.
In your loop, when you assign new data to inputs and targets with each iteration, the old data on the GPU is replaced. If there are no other references to the old data, PyTorch marks that memory as free. However, "free" in PyTorch's allocator means it's available for future allocations by PyTorch operations, not that it has been returned to the GPU's free memory pool.


Direct GPU Initialization: Should show an immediate increase in GPU memory usage as the data is generated.
CPU to GPU Movement: Will show an increase in CPU memory usage initially and then a jump in GPU memory usage when the data is transferred.


how is cuda memroy managed:? https://stackoverflow.com/questions/8684770/how-is-cuda-memory-managed
https://stackoverflow.com/questions/64068771/pytorch-what-happens-to-memory-when-moving-tensor-to-gpu
https://stackoverflow.com/questions/63061779/pytorch-when-do-i-need-to-use-todevice-on-a-model-or-tensor

https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
