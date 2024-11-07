# Update - memory usage

## To do before next meeting:

- smaller learning rate - so that accuracy with smaller batches go up.
- reserved memory amount - in pytorch
    - How does memory allocation work in Pytorch?
    - How does memory allocation work in Pytorch on the jetson board?
- CPU and GPU memory sharing - how does it work on the board?
- Where memory is going? (out of 2GB load, 900MB unaccounted for)
    - Trigger garbage collection more often in pytorch.
    - Use latest version of pytorch.
    - Understand memory allocation in Jetson board.
    - Profile memory: check if Valgrind can be used.
    [http://cs.ecs.baylor.edu/~donahoo/tools/valgrind/](http://cs.ecs.baylor.edu/~donahoo/tools/valgrind/)

## Memory tracking:

- Running training on lenet with 128 batch size.
    - Why Lenet? Because I wanted to observe memory usage across one full run without it being killed. This was the lightest. Valgrind itself would occupy some memory and it was being killed for other ones.
- Memory usage measured using different tools:
    - jtop (nano specific)
    - tegrastats (nano specific)
    - htop
    - smem (alternative to htop)
    - Valgrind (measured for training script)
    - Python [memory profiler](https://pypi.org/project/memory-profiler/) (Some of the measurements at lowest level were opaque. Valgrind does show granular but wanted to know memory change per line) (use a native [Python performance analysis tool](https://stackoverflow.com/q/582336/620382) to decipher _PyEval_EvalFrameDefault) [explanation](https://stackoverflow.com/questions/58852025/what-is-pyeval-evalframedefault)
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled.png)
    
- Base state:
    - 900MB – baseline – gnome shell + vscode + other stuff – without desktop (remote ssh)
    - Cached + Free - available for usage.
    - Note:
        - We can increase swap space to be used from SD card, but it will cause faster degradation of the SD card.
        - We can use LXDE instead of GNOME to cut down on the base memory usage.
    
    jtop memory
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%201.png)
    

smem

![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%202.png)

- Training state:
    - 3.4GB being used. If 900MB for base state, how is the remaining 2.5GB accounted?
        - assuming nothing is adding load.
    - jtop tells us 1.1GB being used by our program.
    - htop mentions 2.5GB at one point but comes down quick.
    - Some variability. What is happening? How to track the memory usage? (Calculations can differ due to difference in estimates of shared part of memory between processes. PSS for smem, other schemes for htop)
    
    During training job:
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%203.png)
    
    htop
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%204.png)
    
    jtop mem
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%205.png)
    
    smem system wide totals
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%206.png)
    
    tegrestats
    
    **RAM 3431/3956MB** (lfb 37x4MB) SWAP 1904/1978MB (cached 2MB) IRAM 0/252kB(lfb 252kB) CPU [8%@1479,5%@1479,99%@1479,10%@1479] EMC_FREQ 5%@1600 GR3D_FREQ 0%@76 APE 25 PLL@41C CPU@41.5C PMIC@50C GPU@43C AO@48.5C thermal@41.5C POM_5V_IN 3014/3014 POM_5V_GPU 0/0 POM_5V_CPU 1044/1044
    
    Determined - consensus that usage was spiking to 3.4GB atleast.
    
    ## What next?
    
    Valgrind mentioned that the usage for the training program was 2.1GB.
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%207.png)
    
    At peak:
    
    ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%208.png)
    
    They point to libcudart.so.10.2.300.
    
    Note:
    
    - **`libcudart`:** This is the CUDA runtime library. It **provides low-level functionality for managing GPU devices**, memory allocation, kernel launches, and other runtime operations necessary for running CUDA-accelerated applications. It is a **fundamental component** of the CUDA toolkit.
    - **cuDNN (CUDA Deep Neural Network library):** cuDNN, on the other hand, is a higher-level library specifically designed for deep neural network operations. It includes optimized implementations of various deep learning primitives, such as convolutions, pooling, activation functions, and more. cuDNN is intended to accelerate deep learning frameworks like TensorFlow, PyTorch, and others by providing highly optimized GPU-accelerated routines.
        - It is needed for making inferences.
        - [Nvidia said](https://forums.developer.nvidia.com/t/problem-with-limited-memory-after-loading-model-jetson-nano/184956) that on jetson boards, cuDNN can add upto unaccounted 600-900MB worth of memory usage while it is running. To resolve inference memory issues, use tensorRT which is a high performing deep learning inference library. It was low memory footprint and is [recommended](https://forums.developer.nvidia.com/t/memory-usage-in-pytorch/185671) for jetson devices.
            - But primarily used in inference and we are training here, so what is happening?
            - Valgrind massif confirms that most of the memory usage is coming from libcudart and not cuDNN.

### Small digression:

- I looked at how torch would need to use memory for training.
- Found a [blog](https://medium.com/deep-learning-for-protein-design/a-comprehensive-guide-to-memory-usage-in-pytorch-b9b7c78031d3), where she had explained:
    - Total maximum memory usage in future iterations will be: model + activations + gradients + gradient moments, which means the memory usage will increase on the second pass but then remain constant. (what I observed in valgrind output)
- Used her code to check the memory being used from pytorch’s perspective.
    - Very small model:
        - I was only interested in quantifying the usage from a known perspective rather than what the model entails. She had recorded footprint of ~37MB.
        - I recorded ~36MB using same code and functions. I noticed that the **torch.cuda.memory_allocated(device)** won’t paint the complete picture in our case. When the device is gpu, it will give memory usage for that only.
        - On the board, there is a shared memory between CPU and GPU. No separate allocation for the GPU.
        - In my other measurements, I was seeing much higher spikes similar to what I saw for lenet so I was hesitant to stop here. I checked using Valgrind and found that 1.4GiB was being used and the callstack was stopping at  **_PyEval_EvalFrameDefault**. Needed more info. I used a python memory profiler to check line by line memory usage. Since, it worked here, I applied it on our main lenet code.
        
        ![Untitled](Update%20-%20memory%20usage%20091f495a9cb8420da997803ba6056df2/Untitled%209.png)
        

### Then What Next?

- We know that our program is taking 2.1Gib ~ 2.25GB. (from Valgrind)
- But where exactly?
    - 87 2081.602 MiB 1585.188 MiB 1 net = [net.to](http://net.to/)(device)
    - Show line by line change - lenet-profile/ profile-out.txt

- Then I checked for: **[TORCH.NN.MODULES.MODULE.to](http://TORCH.NN.MODULES.MODULE.to) method - which is where I am right now.** I need to figure out why its creating this footprint and what to do for jetson nano?
    - [https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.to](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.to)

Notes:

- Pytorch version:
- smem:
    
    Smem -w -k -t
    Smem -m -k -t
    Smem -u -k -t
    K stands for MB/GB, t for total, w for system wide including kernel space, u for all users, sudo needed to get total data
    
- Tools used:
    - Jtop
    - Htop
    - Smem: [https://opensource.com/article/21/10/memory-stats-linux-smem](https://opensource.com/article/21/10/memory-stats-linux-smem)
        - https://www.youtube.com/watch?v=HD3Z7cExFMk
    - Valgrind: [https://stackoverflow.com/questions/24935217/how-to-install-valgrind-properly](https://stackoverflow.com/questions/24935217/how-to-install-valgrind-properly)
        - [https://github.com/KratosMultiphysics/Kratos/wiki/Checking-memory-usage-with-Valgrind](https://github.com/KratosMultiphysics/Kratos/wiki/Checking-memory-usage-with-Valgrind)
        - [https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/performance_tuning_guide/ch05s03s03](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/performance_tuning_guide/ch05s03s03)
- A paper on jetson nano benchmarks: [https://arxiv.org/ftp/arxiv/papers/2307/2307.16834.pdf](https://arxiv.org/ftp/arxiv/papers/2307/2307.16834.pdf)