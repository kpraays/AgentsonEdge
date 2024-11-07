# Agents on Edge? :)

In the months leading up to the summer of 2024, driven by my desire to see the benefits of AI spread to all, I explored the deployment of LLMs on resource-constrained edge devices. Noticing that LLMs are predominantly concentrated in cloud/data centers, I questioned whether those without access to the latest hardware would be left behind. My focus shifted from output quality to the system aspects of deploying LLMs for inference on edge devices.

Recognizing the rising use cases of LLM agents, during the summer, I worked with Nvidia's Jetson series, a platform designed for edge AI and robotics. I investigated the feasibility of running LLM agents in resource-limited environments. This project was both an experiment and a learning journey.

> This blog is structured to guide you through what was done, why it was done, the challenges faced, and how you can do it too. It's divided into three parts: background, implementation, and analysis. The background and implementation sections provide practical insights, while the analysis mirrors the results and findings from my project report.
> 

## *Abstract*

Edge devices are inherently constrained by their computational power, memory, and other resources. Their small form factor, limited power supply, and heat dissipation capabilities make them ubiquitous in today’s tech ecosystem, especially at endpoints close to the user. With advancements in large language models (LLMs) and inference technologies, AI-powered agents are emerging as valuable applications. These agents are expected to perform much of the computational work for AI applications, particularly near user endpoints. This begs the question,  are edge devices suitable candidates for deploying these agents? More specifically, can edge devices handle the machine learning workloads associated with LLM-powered agents? Given the vast array of AI devices, configurations, and software dependencies, arriving at a definitive answer is challenging. However, exploring the feasibility and establishing a baseline is an essential first step in this direction.

## *Introduction to the problem*

- This project explores the feasibility of deploying small language model (SLM) agents on edge devices with older hardware and limited memory. While large language models (LLMs) are commonly hosted in cloud environments, we aim to investigate the potential of self-hosted, small-scale models running on consumer-grade hardware to support local AI agents.
- The challenge, however, lies in determining whether these devices can handle the expected workloads when deploying locally hosted small language models, given the constraints of memory and hardware portability inherent to edge devices.
- For this study, the Nvidia Jetson Nano developer kit was selected as a representative edge device. It’s important to note that this choice imposes certain limitations in terms of memory capacity and software compatibility. The Jetson Nano 4GB module is no longer actively supported with the latest AI tools and libraries from Nvidia, as the newer Jetson Orin Nano is now the current supported offering. This lack of support may lead to challenges in running machine learning and deep learning inference and training packages on the older hardware.

## *Objectives*

- **Deployment Feasibility**:
    - Can a small LLM be successfully deployed on the Nvidia Jetson Nano developer kit (4GB edition) for text-based tasks without rendering the environment unusable for other processes? Additionally, is there enough capacity to add wrapper files for local agents while the LLM executes queries?

- **Workload Analysis**:
    - How do common types of query workloads associated with an LLM agent perform on edge devices? What trends and patterns emerge from these performances, and what are the underlying reasons?

# **Background**

## **Nvidia Jetson Nano and Nano Developer Kit**

The Nvidia Jetson Nano is a compact, powerful, and cost-effective computing module designed for AI and machine learning tasks at the edge. It is part of the Nvidia Jetson platform, which includes a range of modules and developer kits aimed at bringing AI capabilities to embedded systems.

Nvidia offers two types of boards: Developer Kits and Jetson Modules. The developer kit is intended for prototyping and project development. Jetson Modules, on the other hand, are built for deployment in production environments throughout their [operating lifetime](https://developer.nvidia.com/embedded/faq#jetson-lifetime). They undergo full functionality and reliability validation across various environmental conditions, making them suitable for deployment in diverse end-user environments ([more details](https://developer.nvidia.com/embedded/faq#jetson-devkit-not-for-production)).

For this project, we used the Jetson Nano Developer Kit 4GB variant, featuring a 128-core NVIDIA Maxwell GPU, a Quad-core ARM Cortex-A57 CPU, and 4GB of 64-Bit LPDDR4 RAM. As of August 2024, Nvidia offers limited support for this version, having shifted focus to the newer generation, the [Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit), which features 8GB of memory and enhanced power capabilities. The Orin series provides greater memory and power efficiency, marking a significant upgrade from the 4GB memory and 10W max power of the Jetson Nano module.

For a quick introduction to our device, you can watch a video [here](https://www.youtube.com/watch?v=Uvu6NNOvhg4). The official user guide and interface details are available [here](https://developer.nvidia.com/embedded/dlc/Jetson_Nano_Developer_Kit_User_Guide), along with the officially supported components list.

## **Clarifying Nvidia's Terminology**

Nvidia uses several similar-sounding terms in their robotics and edge computing ecosystem, which can benefit from clarification to avoid confusion later in our blog. The Jetson lineup includes various devices—such as AGX Orin, Orin NX, Orin Nano, TX2, and Nano—that are designed to meet different user requirements and cost brackets. Despite the differences in hardware, the Jetson series features a unified embedded AI software [stack](https://developer.nvidia.com/embedded/develop/software) that abstracts the heterogeneous low-level differences in devices and integrates the high-level AI frameworks and tools commonly used in the industry.

## **JetPack SDK Overview**

[JetPack](https://developer.nvidia.com/embedded/jetpack) is a crucial part of the software stack provided by Nvidia for their edge AI devices. The JetPack SDK primarily includes:

1. **Jetson Linux**: A reference file system derived from Ubuntu, containing a Board Support Package (BSP) with a bootloader, Linux kernel, Ubuntu desktop environment, and NVIDIA drivers.
2. **AI Stack**: A CUDA-accelerated AI stack that includes:
    - **NVIDIA TensorRT and cuDNN** for accelerated AI inference.
    - **CUDA** for accelerated general computing.
    - **VPI** for computer vision and image processing.
    - **Jetson Linux APIs** for multimedia acceleration.
    - **libArgus and V4L2** for camera processing.
    
    The availability of these packages depends on the JetPack version compatible with your Jetson device. Compatibility can be checked using the [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive). Depending on the end-of-support timelines, different devices may be locked to specific JetPack versions, meaning the OS version, libraries, and dependencies will be limited to those available in the latest supported JetPack version. For example, the Jetson Orin Nano is expected to be supported until 2030 based on the hardware roadmap, minimizing the risk of dependency issues due to obsolescence ([roadmap](https://developer.nvidia.com/embedded/develop/roadmap)).
    
3. **Developer Tools**:
    - **Development and Debugging**: Tools such as Nsight Eclipse Edition, CUDA-GDB, and CUDA-MEMCHECK.
    - **Profiling and Optimization**: Tools including Nsight Systems, nvprof, Visual Profiler, and Nsight Graphics.

## **Unified Memory Management on Jetson Nano**

The Jetson Nano board features a unified memory management system, where the CPU and GPU share a common pool of DRAM. This architecture allows memory allocated by the CPU to be directly accessed by the GPU, and vice versa. However, special care is required when working with PyTorch GPU code, as it typically does not account for this unified memory architecture. This can lead to data being copied twice in memory—once on the CPU side and once on the GPU side—resulting in inefficient memory usage. For more details, you can refer to discussions [here](https://discuss.pytorch.org/t/pytorch-with-cuda-unified-memory/60783) and [here](https://ieeexplore.ieee.org/document/9599280).

## **Why small LLMs?**

Given that our target deployment device is constrained to 4GB of shared memory (GPU + CPU), traditional large language models (LLMs) are not ideal candidates for local deployment. Small language models (SLMs), which are scaled-down versions with significantly fewer parameters—ranging from a few hundred million to a few billion—offer a more suitable alternative. While these models may lack the vast parameter size of full-scale LLMs, they compensate with a much smaller footprint on system resources.

Small LLMs can be fine-tuned for specific tasks while ensuring lower computational requirements, reduced memory usage, and faster inference times. They also consume less power during both training and inference, making them particularly valuable for edge devices. This balance of performance and efficiency, combined with their ease of fine-tuning and deployment for niche applications, makes them adaptable to various environments for a broad range of applications.

**Memory footprint for common small LLMs on Orin series Jetson modules**

(source: Jetson AI lab SLM [benchmarks](https://www.jetson-ai-lab.com/tutorial_slm.html))

| Small Language Models (4-bit Quantization) |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |
| **Name** | **Base Model** | **Chat Model** | **Params** | **ContextLength** | **Open LLMAvg Score** | **Tokens per Second** |  | **Memory** |
|  |  |  |  |  |  | **Orin Nano** | **AGX Orin** |  |
| StableLM-1.6B | [stabilityai/stablelm-2-1_6b](https://www.google.com/url?q=https://huggingface.co/stabilityai/stablelm-2-1_6b&sa=D&source=editors&ust=1717566531684685&usg=AOvVaw3f_ySgYaitZYFVkucJAGm2) | [stabilityai/stablelm-2-zephyr-1_6b](https://www.google.com/url?q=https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b&sa=D&source=editors&ust=1717566531684836&usg=AOvVaw3J60QlBDD8msJoXjYmUxZu) | 1.6B | 4096 | 45.25 | 36.8 | 110.5 | 1029 MB |
| Phi-2 (2.7B) | [microsoft/phi-2](https://www.google.com/url?q=https://huggingface.co/microsoft/phi-2&sa=D&source=editors&ust=1717566531685632&usg=AOvVaw2bBU3yx7wAi_ZSdTv97Ruk) | [microsoft/phi-2](https://www.google.com/url?q=https://huggingface.co/microsoft/phi-2&sa=D&source=editors&ust=1717566531685720&usg=AOvVaw1Zw2FL7S-aIyQL7BrCy5NF) | 2.7B | 2048 | 61.33 | 24.1 | 73.5 | 2210 MB |
| Gemma-2B | [google/gemma-2b](https://www.google.com/url?q=https://huggingface.co/google/gemma-2b&sa=D&source=editors&ust=1717566531686057&usg=AOvVaw2z9ILjLtSVgnOZtnCHMz93) | [google/gemma-2b-it](https://www.google.com/url?q=https://huggingface.co/google/gemma-2b-it&sa=D&source=editors&ust=1717566531686122&usg=AOvVaw27yaRLXjDXEf3B0rHjWyQK) | 2.5B | 8192 | 46.37 | 26.9 | 75.1 | 2037 MB |
| TinyLlama-1.1B | [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://www.google.com/url?q=https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T&sa=D&source=editors&ust=1717566531686419&usg=AOvVaw14Ti4D9C_Bvv05-jsZAH5m) | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://www.google.com/url?q=https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0&sa=D&source=editors&ust=1717566531686481&usg=AOvVaw1PKvqi2Q7MV6xs6mhReLCr) | 1.1B | 2048 | 37.17 | 67.6 | 150.2 | 622 MB |
| ShearedLlama-1.3B | [princeton-nlp/Sheared-LLaMA-1.3B](https://www.google.com/url?q=https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B&sa=D&source=editors&ust=1717566531686770&usg=AOvVaw1V_0Gjy3WiyKWWXrqNYCzE) | [princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT](https://www.google.com/url?q=https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT&sa=D&source=editors&ust=1717566531686828&usg=AOvVaw3GXzQRfvHow0-f33PMdO9k) | 1.3B | 4096 | 37.14 | 56.3 | 146.2 | 828 MB |
| OpenLlama-3B | [openlm-research/open_llama_3b_v2](https://www.google.com/url?q=https://huggingface.co/openlm-research/open_llama_3b_v2&sa=D&source=editors&ust=1717566531687437&usg=AOvVaw1UQBK-nDYsrh2gtq3gc8QA) | [openlm-research/open_llama_3b_v2](https://www.google.com/url?q=https://huggingface.co/openlm-research/open_llama_3b_v2&sa=D&source=editors&ust=1717566531687507&usg=AOvVaw19g80rZ9xBnG77YDxLWeZg) | 3B | 2048 | 40.28 | 27.3 | 81.9 | 2246 MB |
| Llama2-7B | [meta-llama/Llama-2-7b-hf](https://www.google.com/url?q=https://huggingface.co/meta-llama/Llama-2-7b-hf&sa=D&source=editors&ust=1717566531687795&usg=AOvVaw1McltMN4EkepulZ8yA6mTj) | [meta-llama/Llama-2-7b-chat-hf](https://www.google.com/url?q=https://huggingface.co/meta-llama/Llama-2-7b-chat-hf&sa=D&source=editors&ust=1717566531687863&usg=AOvVaw3I28OC8ii9MdoUZiV9XAqf) | 7B | 4096 | 50.74 | 16.4 | 47.1 | 4298 MB |

## TinyLlama

The TinyLlama project involved the pretraining of a 1.1B parameter Llama model on 3 trillion tokens using 16 A100-40G GPUs. By utilizing the same architecture and tokenizer as Llama 2, TinyLlama is highly compatible with many open-source projects built on the Llama framework. For our workloads, we used the model [TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf](https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF), a 5-bit GGUF format version of the [TinyLlama 1.1B Chat v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model. This model was fine-tuned on top of [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) using [Hugging Face’s Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) training recipe.

### Why was TinyLlama considered?

TinyLlama was chosen for deployment on the Nano board because:

- **Efficient Model Size**: With 1.1 billion parameters, TinyLlama is an efficient variant of the larger LLaMA architecture, making it suitable for resource-constrained environments.
- **Quantized Versions Available**: The availability of quantized versions of the model allows for reduced memory usage.
- It is part of the **open-source** AI community.
- **Tooling Support:** It is supported by tools and frameworks like llama.cpp.

## Agents

![Components of an LLM agent ([source](https://lilianweng.github.io/posts/2023-06-23-agent/#citation))](Untitled.png)

Components of an LLM agent ([source](https://lilianweng.github.io/posts/2023-06-23-agent/#citation))

LLM agents ([Weng](https://lilianweng.github.io/posts/2023-06-23-agent/#citation)) extend the capabilities of language models beyond simple text generation. They have the following components:

1. **Planning**: ****deals with subgoal decomposition and refinement. The LLMs serve as the planning module, determining the next actions based on the current state of the system, which is provided as part of the prompt. An intermediary wrapper captures the system state at each stage, feeding it back into the LLM to generate subsequent actions.
2. **Memory**: The state recorded during interactions, along with external knowledge, is stored in the memory module. Long term memory can be implemented by storing external information in databases. Short term memory means keeping a track of the goals of the current interaction (using vector databases to store embeddings of user utterances is an example)
3. **Tool Use**: LLM agents interact with external APIs to perform tasks.

## **Why LLM Inference on Edge is significant?**

Performing inference directly on an edge device, such as the Jetson Nano, offers several significant advantages:

- **Reduced Latency**: By processing data locally, there’s no need for it to travel to a central server or cloud. This drastically reduces response times, which is crucial for applications requiring quick decision-making or real-time feedback. Examples include autonomous vehicles, real-time language translation devices, and interactive customer service kiosks.
- **Privacy and Security**: Local data processing ensures that sensitive information remains on the device, which is critical for maintaining confidentiality in sectors like healthcare and finance. This is particularly important when dealing with potentially sensitive user data.
- **Reliability and Availability**: Edge devices can operate independently of the cloud, meaning they continue functioning even during network outages. This reliability is vital for remote monitoring systems in agriculture, healthcare in isolated locations, and industrial IoT applications where consistent operation is necessary.
- **Cost-Effectiveness**: Reducing dependency on cloud computing resources for every inference task can significantly cut down on data transmission costs and cloud service fees.
- **Compliance with Regulations**: Local data processing aids in compliance with regulations like GDPR in Europe, which impose strict rules on data transfer and storage, particularly when handling personal data across borders.

# **Implementation pointers**

## **Setting up Jetson Nano**

---

### **Getting the Board Up and Running: Power Supply and Hardware Considerations**

To run the Jetson Nano board independently, without relying on a PC or phone power bricks, we needed to procure the following hardware:

- **5V 4A (4000mA) Switching Power Supply with 2.1mm DC Barrel Connector**: Nvidia has tested and recommended [this one](https://www.adafruit.com/product/1466).
- **SanDisk Ultra 128GB microSDXC UHS-I Card**.
- **Micro USB Cable**.
- **Jumpers (x1)**: [link](https://www.amazon.ca/ZYAMY-2-54mm-Standard-Circuit-Connection/dp/B077957RN7).
- **Wi-Fi Adapter**: [link](https://www.amazon.ca/Geekworm-Adapter-Wireless-1200Mbps-Network/dp/B07TFT876R).

The Jetson Nano module requires a minimum of 4.75V to operate, which can be supplied via the Micro-USB connector. Typically, micro-USB cables are rated to deliver around 5V. However, using the board in headless mode requires a DC power supply, meaning the board needs its own set of peripherals instead of being controlled through a PC. The micro-USB port on the board can only function in one mode at a time—either to power the board or to connect with a PC. When connected via the micro-USB port, the DHCP server is automatically activated on the board to help assign IP addresses.

If the total load on the developer kit is expected to exceed 2A, such as when peripherals are attached to the carrier board, you will need to use jumpers to connect the J48 Power Select Header pins, disabling power supply via Micro-USB. This enables 5V/4A power via the barrel jack connector, allowing you to use the power adapter (such as Adafruit’s GEO241DA-0540 [Power Supply](https://www.adafruit.com/product/1466)).

The total power usage of the developer kit is the sum of the power consumed by the carrier board, module, and peripherals. The carrier board consumes between 0.5W (at 2A) and 1.25W (at 4A) with no peripherals attached. Additionally, the Jetson module supports two software-defined power modes, which constrain the module to 10W or 5W budgets by capping the GPU and CPU frequencies and the number of active CPU cores. The power supply used should align with the expected power budget based on your use cases. For all workloads in this project, the 10W mode was used, along with the power adapter, to supply power to the board. Commands were sent and data recorded using a SSH connection over micro-USB to a laptop.

**Tips:**

<aside>
💡 **Dedicated Micro-USB Cable**: Be mindful of potential undercurrent/overcurrent warnings during processing (for example, after the first boot-up or when cloning a repo from GitHub). This issue can sometimes be mitigated by shortening the length of the power cable used for micro-USB. Using a dedicated power source instead of a laptop’s USB port is advisable. Monitoring tools like `tegrastats` may show that power consumption (POM_5V_IN) occasionally exceeds the upper limit.

</aside>

<aside>
💡 **Dedicated DC Power Supply**: It’s recommended to have a dedicated DC power supply for each board, with the jumper in place, to free up the micro-USB port.

</aside>

### Flashing the Board

Before running workloads on the Jetson Nano board, it needs to be flashed with the Jetson Nano Developer Kit image. Nvidia provides this [image](https://developer.nvidia.com/jetson-nano-sd-card-image) on their "Getting Started with Nano" [page](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro).

1. First we prepare the SD card for flashing by a “quick format” using SD card formatter [tool](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/) then we write the Jetson Nano Developer Kit SD Card Image to the microSD card using [etcher](https://etcher.balena.io/). 

<aside>
💡 Ensure the lock switch on the SD card adapter is in the correct position. If it's locked, the flash process will fail even though formatting may work. The locked status will also be reflected in Etcher.

</aside>

[https://lh7-us.googleusercontent.com/docsz/AD_4nXd6rjszrk5SNG20n5f7lVV3zh4JX6oY8P0aiBw9qpiovWxwUShc51uDVSpiT3RaaRYLuIhTn8G20Q2llB_KUiPbjpBpOOxCLrjROkuWssCMOEL9jJnNmXE1Nw-Rd0VuMzMpcpmcrYfDc0eDTeQmgxbRgfPl?key=BwhS4FIqQ0c1QLGcIRCklA](https://lh7-us.googleusercontent.com/docsz/AD_4nXd6rjszrk5SNG20n5f7lVV3zh4JX6oY8P0aiBw9qpiovWxwUShc51uDVSpiT3RaaRYLuIhTn8G20Q2llB_KUiPbjpBpOOxCLrjROkuWssCMOEL9jJnNmXE1Nw-Rd0VuMzMpcpmcrYfDc0eDTeQmgxbRgfPl?key=BwhS4FIqQ0c1QLGcIRCklA)

1. For the initial setup after flashing the microSD card, it’s often easier to directly connect a monitor and peripherals to the board and follow the on-screen instructions for getting started.
    - After completing the initial setup, you can SSH into the board and work remotely, which I found to be the fastest way to get up and running.
    - Alternatively, the initial setup can be done in headless mode (without a display attached).

![The SD card goes in the slot under the heat sink. Source: [get-started-jetson-nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup)](image.png)

The SD card goes in the slot under the heat sink. Source: [get-started-jetson-nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup)

### **Software Stack and Compatibility**

When we flash the Jetson Nano Developer Kit image, the JetPack SDK is also loaded onto our module. The Jetson Nano 4GB supports up to the JetPack 4.6 release, specifically [JetPack SDK 4.6.5](https://developer.nvidia.com/jetpack-sdk-465). As of August 2024, JetPack 6.0 is the current production release, available in the JetPack [archive](https://developer.nvidia.com/embedded/jetpack-archive).

This means our Jetson module will run [Jetson Linux 32.7.5](https://developer.nvidia.com/embedded/linux-tegra-r3275), which is based on Ubuntu 18.04. This version is important to consider when addressing dependencies for our workloads. The module includes CUDA 10.2, cuDNN 8.2, TensorRT 8.0, and GCC 7. The version of PyTorch compatible with our Nano module is 1.10, as version 1.11 and above requires Python 3.7, which is introduced in JetPack 5.0. Additionally, PyTorch 2.0 and above uses CUDA 11, available in higher versions of JetPack. According to Nvidia developer forums, independent updates of CUDA, cuDNN, or TensorRT versions outside of JetPack are [not supported](https://forums.developer.nvidia.com/t/cuda-11-2-installation-on-jetpack-4-6-2/216992/5).

The wheels for JetPack 4.6 configuration with Python 3.6 have been built by Qengineering and are available [here](https://github.com/Qengineering/PyTorch-Jetson-Nano). Installation commands for PyTorch version 1.10 are also available [here](https://qengineering.eu/install-pytorch-on-jetson-nano.html). (A local copy of the wheel is stored as a backup in case the GDrive link becomes unavailable.)

<aside>
💡 Attempting to sideload a higher version of PyTorch onto the Jetson module may result in the model running in CPU-only mode, which was not the goal for this project. Therefore, checking dependency requirements is crucial.

</aside>

<aside>
💡 Nvidia developer forum references:
- [Pytorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) - summarised versions according to Jetpack support
- Pytorch is CPU only version - Jetson Nano Jetpack 4.6: [here](https://forums.developer.nvidia.com/t/installing-pytorch-on-jetson-nano/222007/4) and [here](https://forums.developer.nvidia.com/t/how-to-enable-cuda-with-pytorch-running-on-a-jetson-nano-2gb-device/282762/3)
- Jetpack incompatibility: [here](https://forums.developer.nvidia.com/t/cant-install-pytorch-on-jetsonnano-p3450/262244/3)

</aside>

[data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)

### Connecting the Board to the Internet

To connect the Jetson Nano module to the internet, additional drivers may be required to get the Wi-Fi adapter working. Supported Wi-Fi adapters can be determined from this [table](https://elinux.org/Jetson/Network_Adapters).

- For the rtl88x2bu Wi-Fi adapter, the driver is available on this GitHub [repository](https://github.com/cilynx/rtl88x2BU_WiFi_linux_v5.3.1_27678.20180430_COEX20180427-5959).
- For TP-Link adapters with the Realtek RTL8821AU chipset ([source](https://github.com/nlkguy/archer-t2u-plus-linux)), use this [driver](https://gist.github.com/Tiryoh/92579a55eba17286855872bcc7fafebd) or [this GitHub repository](https://github.com/aircrack-ng/rtl8812au).

### Ways to Connect Jetson Nano to a PC

- **SSH Over Micro-USB** or Direct LAN or Wi-Fi:
    - For my work, I used remote SSH to the board over micro-USB, using VS Code and WinSCP for file transfer.
    - The IP address for SSH is “192.168.55.1”.
    - Over IPv4, you can connect to only one board per PC, but with IPv6, you can connect to multiple boards on the same PC.
- **VNC**: Access the desktop GUI through a host (similar to TeamViewer).
- **Serial Over USB**: Access the board using a host terminal.

### Why was LLM inference chosen as workload?

Our initial goal was to leverage the NVIDIA Jetson Nano's capabilities to train deep learning models directly on the device. The Jetson Nano, equipped with 4GB of shared memory between the GPU and CPU, presents unique challenges due to this configuration. Our exploration involved models commonly used in image processing tasks, such as ResNet and MobileNet.

**Memory Challenges**

We encountered out-of-memory (OOM) errors that hindered on-device training. The shared memory architecture, while cost-effective, complicates data handling as data loaded onto the GPU is duplicated, consuming valuable memory resources. This issue was compounded by the device's limitations in supporting post-2022 bindings and libraries for deep learning, restricting our ability to utilize the latest software optimizations.

**Framework Considerations**

We considered modifying the PyTorch framework to improve memory allocation efficiency. However, the complexity of these changes and the modest expected improvements led us to conclude that this approach was not feasible given the fixed hardware capabilities and compatibility issues with newer library versions.

So we determined that the device is better suited for inference instead of training. 

## Deploying the Model

For deploying models on the Jetson Nano module, the following approaches were considered for LLM inference:

1. **Using Llama.cpp**: This approach was preferred due to the stated goal of the llama.cpp project, which is to "enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware." Since we aimed to deploy TinyLlama 1.1B without additional dependencies and with the least possible memory overhead, llama.cpp was our first choice.
2. **Using TensorRT**: This approach is expected to deliver better performance on Nvidia edge GPUs but requires exporting the model to ONNX format before running inference, which adds complexity to the process.
3. **Using PyTorch Directly**: Running queries directly in PyTorch was considered, but the memory overhead due to the unified memory architecture and the need to create a wrapper for managing input to the model were significant drawbacks.

Given these considerations, [llama.cpp](https://github.com/ggerganov/llama.cpp) was chosen as the method for running inference queries on the Jetson module.

<aside>
💡 During later stages of the project, we discovered [LlamaEdge](https://github.com/LlamaEdge/LlamaEdge), which promises even greater efficiency than llama.cpp for inference with only a 30MB overhead. LlamaEdge is designed to be hardware-agnostic and supports all Large Language Models (LLMs) based on the Llama2 framework.

</aside>

As mentioned earlier, the software stack on JetPack 4.6.x is not the most compatible with recent machine learning developments. Due to its limitation at CUDA 10.2 (and Python 3.6, along with other previously mentioned packages), careful attention is needed to resolve dependency errors when running the Llama.cpp inference runtime. This means that direct pip installation of llama.cpp is not possible. Since there is no pre-built image, we will need to build our own image for the board.

We referenced the steps provided by Flor Sanders [here](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f) and made certain adjustments based on the issues encountered during deployment on our Jetson module. Specifically, we need to build llama.cpp for the board, which requires the GCC 8.5 compiler. By default, GCC 7 is shipped with JetPack 4.6.x, so the first step is to build the GCC 8.5 compiler for the board.

### **llama.cpp and gcc 8.5 steps**

1. Download and build the gcc 8.5 compiler.
    - Set up the GCC and G++ environment variables so that Lllama makefile can use these.
    - You can check if the install is working fine using version commands.

```bash
wget https://bigsearcher.com/mirrors/gcc/releases/gcc-8.5.0/gcc-8.5.0.tar.gz
sudo tar -zvxf gcc-8.5.0.tar.gz --directory=/usr/local/
cd /usr/local/
./contrib/download_prerequisites
mkdir build
cd build
sudo ../configure -enable-checking=release -enable-languages=c,c++
make -j6
make install
```

```bash
export CC=/usr/local/bin/gcc
export CXX=/usr/local/bin/g++
```

```bash
gcc --version
g++ --version
```

<aside>
💡 Building the GCC 8 compiler on the Jetson Nano board will take approximately 4-5 hours. Ensure you use a larger SD card (128GB) because the base OS image, will occupy 15-20GB of space. You will want to have more than half the space available for models and experiments. (You can clear the build files after the GCC compiler is installed to free up space.)

</aside>

1. Building and installing llama.cpp.
    - We will have to clone the llama.cpp repository and roll back to a known working commit which was compatible with the board. (as llama.cpp is an active project with new features being added and somewhere along the way - dependencies will break)
    
    ```
    git clone git@github.com:ggerganov/llama.cpp.git
    git checkout a33e6a0
    ```
    
    - In the `Makefile` for llama.cpp, we had to make the following changes:
    1. **Change `MK_NVCCFLAGS`**: Set `maxrregcount=80`.
    2. **Modify `MK_CXXFLAGS`**: Remove the `mcpu=native` flag.
    
    You can use a patch file to apply these changes quickly. Create the patch file with the necessary modifications and apply changes using `git apply --stat file.patch`
    
    ```
    diff --git a/Makefile b/Makefile
    index 068f6ed0..a4ed3c95 100644
    --- a/Makefile
    +++ b/Makefile
    @@ -106,11 +106,11 @@ MK_NVCCFLAGS = -std=c++11
     ifdef LLAMA_FAST
     MK_CFLAGS     += -Ofast
     HOST_CXXFLAGS += -Ofast
    -MK_NVCCFLAGS  += -O3
    +MK_NVCCFLAGS += -maxrregcount=80
     else
     MK_CFLAGS     += -O3
     MK_CXXFLAGS   += -O3
    -MK_NVCCFLAGS  += -O3
    +MK_NVCCFLAGS += -maxrregcount=80
     endif
    
     ifndef LLAMA_NO_CCACHE
    @@ -299,7 +299,6 @@ ifneq ($(filter aarch64%,$(UNAME_M)),)
         # Raspberry Pi 3, 4, Zero 2 (64-bit)
         # Nvidia Jetson
         MK_CFLAGS   += -mcpu=native
    -    MK_CXXFLAGS += -mcpu=native
         JETSON_RELEASE_INFO = $(shell jetson_release)
         ifdef JETSON_RELEASE_INFO
             ifneq ($(filter TX2%,$(JETSON_RELEASE_INFO)),)
    ```
    

- The CUDA toolkit comes pre-installed with the JetPack image we flashed earlier. The necessary files should be located in `/usr/local/cuda` by default. Add them to the path so that nvcc can be used directly.

```bash
$ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- We need to know the compute capability of our GPU. It will need to be set as one of the flags to compile llama.cpp so that it can use GPUs during inference. The following snippet can be used to determine it as we do not have nvidia-smi for jetson module.

```c
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
int device;
cudaDeviceProp prop;

cudaGetDevice(&device);
cudaGetDeviceProperties(&prop, device);

printf("Compute capability: %d.%d\\n", prop.major, prop.minor);

return 0;
}
```

```bash
/usr/local/cuda/bin/nvcc compute_capability.cu -o compute_capability
```

**Compute capability for our Jetson module is 5.3.**

- We then build the llama.cpp using:

```bash
make LLAMA_CUBLAS=1 CUDA_DOCKER_ARCH=sm_53 -j 6
```

### Failures Observed

1. **Cross-Compiling Llama.cpp**:
    
    Given the slow process of building the GCC 8.5 compiler directly on the Jetson Nano, we attempted to cross-compile llama.cpp on a different Unix machine (Intel x64). This approach involved setting up the ARM cross-compiler on the host machine, copying the necessary libraries and headers from the Jetson Nano to the host, and configuring GCC with sysroot support to utilize those libraries and headers.
    
2. **Challenges with Cross-Compilation**:
    
    Although the GCC compiler was successfully built on the host machine, the makefile for llama.cpp required many different environment variables that were challenging to configure. After encountering several errors, we concluded that building both the compiler and llama.cpp directly on the Jetson Nano using Flor Sanders's steps would be more efficient than resolving the cross-compilation issues. While it is theoretically possible to transfer the built GCC compiler to the Jetson Nano, the complexity of setting up the environment led us to revert to building directly on the board. The commands used during our cross-compilation attempt are documented below for reference.
    

Assuming source files for gcc have been downloaded on the host machine.

```bash
cd gcc-8.5.0
./contrib/download_prerequisites
sudo apt-get install gcc-aarch64-linux-gnu
sudo apt-get install binutils-aarch64-linux-gnu
mkdir build-gcc
cd build-gcc
```

```bash
../configure \
--target=aarch64-linux-gnu \
--prefix=/opt/gcc-8.5-aarch64 \
--with-sysroot=/opt/sysroot \
--with-as=/usr/bin/aarch64-linux-gnu-as \
--with-ld=/usr/bin/aarch64-linux-gnu-ld \
--enable-languages=c,c++ \
--disable-multilib

make -j$(nproc)
sudo make install
```

Setting following environment variables can make trying out these commands easier.

```bash
export PATH=/opt/gcc-8.5-aarch64/bin:$PATH
export SYSROOT=/path/to/arm-sysroot
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
```

### **How to run inference queries?**

1. The TinyLlama model can be downloaded using:

```bash

wget https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf
```

1. Queries can be executed using the standard llama.cpp invocation (argument list can be found using the —help tag with the invocation). An example being:

```bash
/nano-llama/llama.cpp/main -m /nano-llama/llama.cpp/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf -p "You are a very helpful AI assistant who gives to the point responses to the user: Hi, how are you today?”
```

## Measuring Readings

To determine the usage of resources during query execution, we utilized the following two packages:

### **Jetson-stats**

The `nvidia-smi` command is not supported on Tegra platforms, including our Jetson Nano module, because Jetsons do not have a discrete GPU on a PCI slot ([source](https://forums.developer.nvidia.com/t/nvidia-smi-not-present-in-jetson-linux/239757/2)). Instead, the [tegrastats](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html#) command is used to report statistics about the device.

However, we found that [Jetson-stats](https://github.com/rbonghi/jetson_stats) is particularly useful for recording measurements. Jetson-stats is a package for monitoring and controlling all devices within the Nvidia Jetson ecosystem. It can be used as a standalone tool (`jtop`) or integrated into Python scripts for API calls. Notably, it does not require `sudo` permissions to determine stats. It’s an open-source project with comprehensive documentation available [here](https://rnext.it/jetson_stats/).

We used Jetson-stats to record:

- General system resource usage.
- Process-wise resource usage.

### **psutil**

[psutil](https://psutil.readthedocs.io/en/latest/) is a cross-platform library for system monitoring, profiling, limiting process resources, and managing running processes. We used psutil to record memory usage and `cpu_percent` utilization by the inference processes.

## Queries for workload

To effectively run our experiments, we curated a dataset comprising 100 queries for each of the five categories. These queries were selected from established NLP datasets commonly used for large language models (LLMs) and aligned with the specific category types.

**Simple Queries**:

- **Dataset**: WikiQA, a question-answering corpus from Microsoft ([*Wiki_qa*](https://huggingface.co/datasets/microsoft/wiki_qa)).
    - Example Queries:
        - "How does interlibrary loan work?"
        - "Where is money made in the United States?"
        - "How does a dim sum restaurant work?"

**Complex Queries**:

- **Dataset**: SQuAD v2.0 (Dev set), a benchmark dataset for machine comprehension of text ([*The Stanford Question Answering Dataset*](https://rajpurkar.github.io/SQuAD-explorer/)).
    - Example Queries:
        - "What does the private education student financial assistance help current high school students who are turned away do?"
        - "Whose infected corpse was one of the ones catapulted over the walls of Kaffa by the Mongol army?"

**Conversational Queries**:

- **Dataset**: Friends Dataset, containing speech-based dialogue from the Friends TV sitcom, extracted from the SocialNLP EmotionX 2019 challenge ([Michelle Li](https://huggingface.co/datasets/michellejieli/friends_dataset)).
    - Example Queries:
        - "Oh, that's right. It's your first day! So, are you psyched to fight fake crime with your robot sidekick?"
        - "Whoa!! Now look, don’t be just blurting stuff out. I want you to really think about your answers, okay?"

**Task-Oriented Queries**:

- **Dataset**: MultiWOZ 2.2, a multi-domain Wizard-of-Oz dataset for task-oriented dialogue systems.We retrieved only the user utterances from the conversation per task and combined them for the model to infer and give the steps in accomplishment of the task ([salesforce](https://github.com/salesforce/DialogStudio/tree/main/task-oriented-dialogues/MULTIWOZ2_2)).
    - Example Query:
        - "This is a bot helping users to find a restaurant and find an attraction. Given the dialog context, please generate a relevant system response for the user: <USER> Hi, could you help me with some information on a particular attraction? <USER> It is called Nusha. Can you tell me a little bit about it? <USER> Thank you. Please get me information on a particular restaurant called Thanh Binh. <USER> Thanks, please make a reservation there for 6 people at 17:15 on Saturday."

**Contextual Queries**:

- **Dataset**: Validation set from Question Answering in Context (QuAC), a dataset for modeling, understanding, and participating in information-seeking dialogue ([*Question Answering in Context*](https://quac.ai/)).
    - Example Query Structure:
        - ::context:: (text provided) ::question:: "Were they ever in any other TV shows or movies?"

# Experiments

1. **Category-Wise Query Submission**:
    - **Process**: We submitted 100 queries from each category sequentially. Every 0.5 seconds, the following metrics were recorded:
        1. **CPU and GPU Memory Usage, CPU Utilization**: These metrics were recorded (across all four cores: CPU utilization) for the query process executed through llama.cpp for TinyLlama.
        2. **System-Wide GPU Utilization**: GPU utilization was monitored while the query execution process was in progress. During data processing, the GPU utilization values used for plots were those that coincided with the same time periods for which per-process readings existed.
        3. **Per Query Metrics**:
            1. **Time to First Token**: The duration before the first token appeared.
            2. **Time Between Tokens**: The time gap observed between the generation of each successive token after the first token had appeared.
            
2. **Granular Query Analysis**:
    - **Process**: We calculated the median length of prompt tokens for each query category and selected one representative query from each category that closely matched the median token length. For each representative query, the same measurements as above were recorded every centisecond (1 centisecond equals 1/100th of a second, corresponding to 100 ticks per second on Ubuntu systems). Recordings were made after a warm-up phase, as discrepancies were found in the CPU usage of complex queries when measurements were taken without warming up. The initial four readings for other queries in each workload were discarded to ensure consistency.

[*Median prompt lengths across categories.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdDsqeY3ClqG_fDQicxfLzuX1zX5xwhWpAcpxSiYTZnc9zvKma5ICHlR-ECoj2nvaa7mI02JmSp27XfI2ZdtCqR8OnpBVB3nt8FHZ_Yv_8o2iWfC8JzjSx7X9oib_-WL_9mbv2owEj0WH-CHvSoSJvlr9wk?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Median prompt lengths across categories.*

**Additional Granular Measurements for Special Scenarios**:

For each of the following scenarios, a warm-up phase was included to ensure consistency in measurements.

1. **Task-Oriented Queries**:
    - **Process**: GPU memory usage was recorded every centisecond for four task-oriented queries. These queries were selected based on an increasing number of tokens in the prompt, with the output token count fixed at 50.
2. **Total Time Measurement**:
    - **Process**: The total time taken to complete two queries was recorded—one from the task-oriented workload and one from the contextual workload. These queries were selected to minimize the difference in the number of tokens in the prompt between them.
3. **Conversational Workload Query**:
    - **Process**: A single conversational query with a number of prompt tokens equal to the category median was selected. GPU memory usage was recorded over five runs of the same query, with the expected output length increasing incrementally in each run. The output token counts were set to ["10", "50", "120", "230", "400"].

# Analysis

## Understanding Resource Usage in LLMs

To understand the trends observed in the experiments defined above, it's important to grasp how outputs are generated by LLMs. 

### **Encoder-Decoder Models**

Encoder-decoder models convert input sequences into latent hidden representations, which they use to generate output responses. This makes them well-suited for tasks like information retrieval and question answering. The process begins with the tokenization of the input prompt, which is then converted into word embeddings based on the model's vocabulary. These embeddings are augmented with positional information, resulting in positional encodings that serve as the input for the encoder blocks. (These embeddings can be moved to CPU memory to free up accelerators for attention operations.)

Within each encoder layer, self-attention operations capture the relationships between tokens in a sentence. This involves calculating how much influence other tokens have on a particular token and scaling the numeric representation of that token by the probability of that influence. To achieve this, three quantities are used: queries (Q), keys (K), and values (V), which are computed by applying weight matrices ($W^Q, W^K, W^V$) to the input X (the encoded representation of the input text sequence) ([LLM inference serving](https://arxiv.org/pdf/2407.12391)).

In models with multi-head self-attention (MHSA), multiple sets of weights for keys and values operate on the same set of tokens per layer. The MHSA block combines features from all attention heads to form its output by multiplying it with another weight matrix, which is also trained with the rest of the system. All these matrices need to be in memory for computation. To maximize the use of hardware accelerators, it's crucial to maintain the state of multiple sets of matrices in the accelerator memory, avoiding transfer delays and idling.

The combined representation of captured self-attention information is normalized and merged back with positional embeddings (residuals) to prevent vanishing and exploding gradients before passing through a feedforward layer. This residual process is repeated with normalization within each encoder block. Different models may have multiple stacked encoder blocks (layers). The final layer provides self-attention values for input tokens, capturing the semantic context, positional ordering, and relationships between tokens in the input.

The decoding phase in an encoder-decoder model utilizes the attention values from the final encoder layer (essentially the higher latent representation of the input data) to generate outputs at each successive decoder layer. This allows the decoder to track significant words in the input. Additionally, because the decoder doesn't know the full output sequence, it applies masked self-attention (explained in decoder section) to record the influence of previous tokens.

### **Decoder-Only Models**

In decoder-only LLMs, output tokens are generally generated sequentially based on the initial input sequence (prompt). These models have a single unit for both encoding the input and generating the output. They use masked self-attention throughout all their layers. The input processing, or prefill stage, is highly parallel, where inputs are embedded and encoded with multi-head attention (steps include tokenization, embeddings, positional embeddings, self-attention values, and normalization with residuals). This stage involves large matrix multiplications that utilize hardware accelerators. Multi-head attention allows the decoder to consider different parts of the sequence in different representational spaces ([LLM inference serving](https://arxiv.org/pdf/2407.12391)).

While calculating self-attention values masked self-attention is used as the decoder is not yet aware about the tokens from the ungenerated sequence. So, only the decoder layers only consider the previous and current tokens when calculating self-attention. This process is repeated over multiple decoder layers until the final decoder outputs are produced. To avoid recomputing the keys and values for the input token while generating the output, the key-value (KV) pairs for previous input tokens can be stored in the accelerator RAM through a KV cache. This cache consumes significant memory during execution but is necessary for maintaining usable inference speed.

The second phase, decoding, involves the autoregressive generation of new tokens by the model, with updates to the KV cache for each new token. The process continues until an `<end of string>` token is encountered or the maximum output length is reached. This is the core part of the inference process.

To convert the decoder layer's values into text, a linear layer is used to map them to the model's vocabulary dimensions, followed by a softmax function to obtain a probability distribution over the vocab tokens, generating the final output.

In its simplest form, the model generates output one token at a time, using the generated tokens along with the existing ones as the input sequence to generate the rest. This approach has limited parallelism compared to encoders.

**Resource Consumption Considerations:**

The encoding phase is highly parallelizable due to independent token processing and the computationally intensive operations on keys, values, and queries. GPUs, with their parallel architecture and dedicated memory, are well-suited for accelerating these computations. However, frequent data transfer between CPU and GPU memory can introduce performance overheads. Consequently, a hybrid approach is often employed, utilizing the GPU for intensive calculations and the CPU for auxiliary tasks and data management. The overall resource consumption of the encoding phase is determined by the model's size, the number of layers, and the computational complexity of the operations performed at each layer.

### TinyLlama: A Specialized Decoder-Only Model

TinyLlama ([TinyLlama: An Open-Source Small Language Model](https://arxiv.org/pdf/2401.02385)) is a decoder-only transformer model based on the architecture of the Llama series. Despite having only 1.1 billion parameters, TinyLlama stands out because it has been trained on an impressive 3 trillion tokens, going against the [Chinchilla](https://arxiv.org/abs/2203.15556) scaling laws. This extensive training dataset allows TinyLlama to achieve significant performance even with a relatively smaller model size.

TinyLlama uses Rotary Positional Embedding (RoPE) ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) to capture positional information. Unlike traditional transformers, where normalization occurs after each transformer layer, TinyLlama, following Llama's architecture, normalizes inputs before each transformer layer using Root Mean Square Layer Normalization ([RMSNorm](https://github.com/bzhangGo/rmsnorm)). RMSNorm has been shown to improve efficiency by 10-50% without a significant decrease in performance ([llama-2-from-the-ground-up](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)).

Similar to the Llama series, TinyLlama uses the SwiGLU activation function instead of the more commonly used ReLU. Additionally, it employs Grouped-query attention, which helps reduce memory bandwidth and speeds up inference. In Grouped-query attention, keys and values can be shared between multiple heads, requiring less memory space than traditional multi-head attention. Specifically, TinyLlama has 32 heads for query attention and 4 groups for key-value heads ([TinyLlama: An Open-Source Small Language Model](https://arxiv.org/pdf/2401.02385)).

### Resource usage contributors

Model size: The number of parameters of an LLM is the approximate total number of weights and biases across all layers that are being used in that model. Thus, the number of weights and the precision in which they are stored will directly influence the space the model takes when loaded in the memory ([Zhang et al.](https://arxiv.org/pdf/2404.14294)).

Attention Operation: As the input length becomes longer, the memory and computational costs associated with attention operation increase quickly because it exhibits quadratic computational complexity in the input length ([Zhang et al.](https://arxiv.org/pdf/2404.14294)).

Decoding Approach: Decoding requires loading KV cache in memory on the accelerator. Though, it happens step by step per token but this cache needs to be in memory for each step and KV cache grows in size as well. This may lead to fragmented memory and irregular memory access patterns ([Zhang et al.](https://arxiv.org/pdf/2404.14294)).

## Analysis of plots across workloads

[*Analysis of CPU and GPU Memory Usage across workloads.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXebf6N7ikqQT9joraEkvZIWiw56IRDB1Z1I9kU3ze5pLoEFFAyQua9YumGQwHReufR0PgzOLZd7FoOp7Yk4heMZ3qgtY7P7G896L2w2MKB_Sp4v-m2SCuV792Zpcop12OvKW0-d0seBr6-gqj4OCELfiqYX?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Analysis of CPU and GPU Memory Usage across workloads.*

### Analysis of Memory usage

We observed that CPU memory usage remains relatively consistent across different workloads, even when the length of queries varies. This consistency is likely due to the storage of embeddings in CPU memory during inference, as these embeddings are required either at the start of the computation for each query or at the end, to convert token IDs back to text from individual probabilities. Since the size of these embeddings does not significantly vary between workloads, the CPU memory usage remains fairly constant. The primary computational load, particularly for attention operations, occurs on the GPU. When the GPU memory becomes overwhelmed, we observe some data being offloaded to CPU memory. This is evident in workloads such as contextual and task-oriented queries, which exhibit higher than average GPU memory usage, accompanied by a slight increase in CPU memory usage, as shown in the zoomed-in version of the CPU memory usage plot.

[*Zoomed in CPU memory usage across workloads.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdWQToxlQ0vU8r9EzMA1RvZcSXkbMmIrNOhaVrvfC7xvfvO77yG_-4Kop0kG6RRtXSp5HrSZYsBdrUlN1XfgCVVqWDSl4XYPPrfdJHXOfiz3_ckAQnxa1q-6SSyJjXH6iCyRn3x_N4N7UCLSoInQC6Jyb9P?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Zoomed in CPU memory usage across workloads.*

Regarding GPU memory usage, the contextual workload stands out with significantly higher memory consumption, while the task-oriented workload shows slight spikes compared to others. This difference can be attributed to the storage of weights, biases, and the KV cache in GPU memory during inference. The KV cache is constructed during the prefill stage and updated with each new token generated during the decoding phase, allowing subsequent tokens to utilize previous KV entries for computing attention. Consequently, the size of the KV cache is directly proportional to the number of tokens, leading to increased memory usage for longer prompts. The average prompt length for queries in the contextual workload is notably longer than in other workloads, with the task-oriented workload following closely behind.

[*Mean prompt length across categories.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdG8nETUTwXJg_5aKV6Hzjd4xpSmvZrMde4DVtPQmx7PpDuENSCfqFqeoHZPRIdNldJ01vDPBdzhtOGuoquYUi8LIFNtwWDxUAbebvlKxHBFfCzUqbeIvnsTG8KvORQ3qYwq846giHveJz9Q7FbB6blexSK?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Mean prompt length across categories.*

### Analysis of Compute

[*Average CPU and GPU usage across workloads.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdlnVLKGLGEGZVe_5FJqQOMeIn_mKWRWYe1c0plGNLfny7EWTuZQYG0zJHE8oEG38QJIhlxKJBJIz1_l1TRKEMtrWiq2cuwPNEZklIB-rIH4Z5ib91Rp2X3jgdkIkcwTP7C2yl8vngmzr818gZwz1ENfgK_?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Average CPU and GPU usage across workloads.*

We observed that average CPU usage is lower for conversational workloads compared to other types. This may be attributed to the TinyLlama model being specifically fine-tuned for chat-related use cases. If conversational patterns were better represented during the model's training stages, the model might be more efficient in handling these workloads, leading to quicker generation times. The near-full GPU utilization observed for conversational workloads could indicate that the GPU is managing these tasks more efficiently, with less need for the CPU to handle intermediate states. Similarly, the simple workload also shows relatively low average CPU usage, likely because these queries are straightforward and recall-based, requiring minimal CPU intervention to manage intermediate states on the GPU.

We also noted significant variation in GPU usage across complex and simple workloads, with noticeable drops in GPU memory usage for these categories, while conversational workloads consistently occupy GPU memory. Fluctuations in GPU usage may occur when the model needs to reference different parts of its knowledge base. Complex queries often demand multiple layers of reasoning and context processing, whereas simple queries typically require straightforward recall, triggering fewer deep transformer layers. This difference is reflected in the GPU usage plot, where the simple workload shows fewer spikes in GPU utilization compared to the complex workload. The same behavior is observed in GPU memory usage. An interesting insight is that some queries in the simple workload may be similar in sentence length and information requirements to those in the complex workload, as suggested by the observed patterns in GPU usage.

The GPU is heavily utilized when numerous parallel operations are required, particularly during token generation (e.g., attention computation, normalization, and feedforward layers—essentially matrix multiplication). The consistently high GPU usage for contextual and conversational workloads indicates that these tasks are more generation-intensive than others. This is intuitive, as the GPU performs significant work in establishing attention with existing tokens in contextual workloads, where queries are generally longer. For conversational workloads, the model's fine-tuning for chat-related queries might result in greater GPU efficiency or utilization, explaining why task-oriented workloads do not exhibit consistently high GPU usage, despite involving step-by-step generation based on user utterances.

Finally, the plots for both memory usage and compute illustrate that different workload types require varying amounts of time to execute, largely due to the expected number of output tokens. Task-oriented workloads, which have an expected output of 100 tokens—the highest among all categories—naturally take longer to generate their output.

***Limits on number of output tokens:***

```python
*{"complex": "70", 
"contextual": "50", 
"conversational": "40", 
"simple": "30", 
"task-oriented": 
"100"}*
```

[*Time to First token (TTFT) and Time between Tokens (TBT) across workloads.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdBoA5EjbDx3qnv--nFOhtp6BS-iOxMtysBOu9oDBkbo3PAeoY-OkU8MZLOGus_3mlltGx02wOiO64QG2XfO9LbEeeNCyOIfHw6T25ah2JDH6fMo8CK2ZcrOGwAqGFZJ274LIvb_OcGYgJuV9C0KjgwEnnC?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Time to First token (TTFT) and Time between Tokens (TBT) across workloads.*

### Latency Analysis

We measured latency in two forms: the time to generate the first output token for a query, referred to as "Time to First Token" (TTFT), and the latency between generating each subsequent token during the decode stage, referred to as "Time Between Tokens" (TBT).

**Time to First Token (TTFT)**:

For workloads with longer prompts, such as contextual and task-oriented queries, we expected TTFT to be greater than for other workloads, which is consistent with our observations. We noticed a larger variation in TTFT for task-oriented workloads compared to others, likely due to the wider range of token counts in the prompts of task-oriented queries. TTFT is heavily influenced by the prefill stage of LLM inference, where the length of the prompt and the time required to build the KV cache directly affect TTFT. An interesting observation is that conversational queries exhibit the least variation in TTFT and have a shorter TTFT compared to simple and complex workloads, despite not being shorter in length. This could be attributed to TinyLlama being specifically fine-tuned for chat instructions, which may enhance its efficiency in handling conversational prompts.

**Time Between Tokens (TBT)**:

TBT essentially represents the time it takes to execute one step of the decoding cycle given the existing sequence. During this stage, the KV cache, which was built and loaded into memory during the prefill stage, must be updated for each new token. Additionally, other operations, such as matrix calculations for normalization and the feedforward layer, are performed. A longer TBT suggests that the GPU is taking more time to complete these operations, potentially indicating that the GPU is compute-bound or that GPU memory is insufficient, leading to data being swapped between CPU and GPU memory, causing delays.

In our analysis, we observed a slight increase in CPU memory usage for contextual and task-oriented workloads, which could suggest some level of data transfer between CPU and GPU memory. However, given that these spikes are minimal, the delays are likely due to the GPU being compute-bound rather than memory-constrained. This indicates that the contextual and task-oriented workloads place a higher strain on the GPU compared to other workloads, as expected.

[*Total time per query vs Tokens in Prompt across workloads.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdFQu0nlDHmRAc8lwJy2aU46CabTUX7Hpg-i08aDRklXmQDh_FAfBxTGMaqPhvAFbG_6vRCc2TeEGyQs3V3-7AcEET91XITW0qN12MoRRLulQyfkABNkiy24L5e8Belp_1Kjtb_WGRexmHVJBXY3OhxZ1s?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Total time per query vs Tokens in Prompt across workloads.*

### Analysis of Total Execution Time for Queries

Task-oriented queries exhibit the largest variation in prompt size compared to other workloads. We anticipated that an increase in the number of tokens in the prompt would result in a longer total execution time for a query, primarily due to the expected increase in TTFT. Additionally, a higher number of expected output tokens would naturally extend the total execution time, as more tokens need to be generated.

In the plot, an increasing trend in "total time for query" is evident within each workload as the number of tokens in the prompt increases. Task-oriented workloads show elevated values for "total time taken for query," which can be attributed to the higher expected output token count in these queries compared to other workloads. However, we also observe variations in the total time taken for queries within the same workload, even when the number of prompt tokens is similar.

These variations may be influenced by how each individual query interacts with the model, particularly in terms of recalling information from deeper layers, reasoning, understanding context, and the GPU's role in generating tokens. The complexity and nature of each query likely contribute to differences in execution time, even within the same workload category.

## Granular analysis of Queries

[*Average CPU and GPU use for a single query.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcuMks2bU6vXiSoKY3cQggGxxEklcTRkyJgf3eYtjCHX6FZ8S5DpN_rQolEObfNaZdySOFDytaY3EUcP1wvDzv6zbujGofefmkSr4cpp1wfF_rf8z7NuL37dKK4wnmaj5bdyMiNm4HXx2TDCYTFvN7t4T4?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Average CPU and GPU use for a single query.*

### Analysis of CPU and GPU Usage Patterns

Our analysis shows that the average CPU usage across different queries remains fairly consistent (excluding measurements from the warm-up phase). However, we observed that task-oriented and contextual queries exhibit two or more GPU usage spikes—typically at the beginning and the end of the query—whereas other query types generally show a spike only at the beginning.

This behavior can be attributed to the nature of these workloads. In contextual queries, the prompt usually includes a paragraph providing context, followed by a question related to that context. For task-oriented queries, the model is asked to function as a bot, assisting the user with specific tasks such as booking movies, hotels, or flights. These queries often involve a sequence of <USER> utterances, requiring the model to understand the task and generate a step-by-step plan for its completion.

As a result, queries in these workloads are generally longer than those in other categories and require the model to deliver output after processing and understanding the context. The multiple GPU usage spikes likely indicate heavy computation occurring in two or more phases. In the first phase, the model might be establishing the context by filling up the KV cache, which is relatively larger than in other workloads. In the second or subsequent phases, the model performs iterative processing to generate responses based on the previously established context.

This pattern is particularly interesting because it suggests that even other workloads (except for simple queries) show smaller subsequent GPU usage spikes, indicating that the model requires initial GPU computation to establish context and additional GPU resources later to generate responses. Although this is a decoder-only model, which typically does not exhibit phase differentiation in GPU usage as encoder-decoder models do, the observed behavior suggests that similar phase-dependent GPU usage patterns may still occur during inference.

[*CPU and GPU memory usage for individual queries.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdJHwNDSxjJfF9RDeW5R19P8Mo-OQGXvXlXuoXmmO7lWak9TdgRlY29ZPshZerRHRtsvbWIVEWMUH4tEMqmn9W_v_AVuHwhOjs941BitACO0xPZfABt8m4aENtO270f2y-E45b3NoPpjbxYi3oPT9aDRmQe?key=7VKbPIlbOp2wxPlxUTA4Ug)

*CPU and GPU memory usage for individual queries.*

### Granular memory usage analysis

**Components Contributing to CPU Memory Usage**

The following components are stored in CPU memory during inference, based on observations from the query logs generated by llama.cpp:

1. **Model Weights**: The model weights are stored in CPU memory. Since these weights are consistent across all query types, this factor remains constant.
2. **Embeddings for Input Tokens**: The embeddings, which map each input token to a high-dimensional vector space, are computed and stored in CPU memory. The embedding layer is the same for all query types, making this a consistent factor across workloads.
3. **KV Cache Allocation**: A fixed portion of the KV cache is allocated in CPU memory, determined by the maximum context length, the number of layers, and the number of heads. This allocation does not vary with the query type.
4. **Pre-Allocated Input and Compute Buffers**: These buffers store intermediate data before it is transferred to the GPU. They remain the same regardless of the query type.

**Components Contributing to GPU Memory Usage**

More complex or lengthy queries, which involve long contexts, more reasoning, and less recall-based output, tend to occupy more GPU memory. This is especially true for contextual, conversational, and task-oriented workloads, as not all queries from complex and simple workloads exhibit these characteristics. The key components taking up GPU memory include:

1. **Temporary Tensor Cores**: GPU memory is required to store temporary tensor cores for operations such as matrix multiplications and attention calculations. Although these tensors are often deallocated after each operation, they can occupy significant space during execution. More complex queries, such as those in contextual or task-oriented workloads, generate more intermediate results, requiring more temporary tensors and thus higher GPU memory usage.
2. **KV Cache**: The KV cache primarily resides in the GPU for faster memory access, supporting attention calculations during the token generation process. While the size of this cache is influenced by the actual context length, it remains fairly consistent across queries, given the absence of an order of magnitude difference in context length.
    - Estimating KV cache usage ([Verma](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)) as:

> **Total size of KV cache in bytes = *2 (two matrices - K,V) * (context length) * (number of heads) * (embedding dimension per head) * (number of layers) *  2 Bytes (sizeof fp16)***
> 

Consider a context length of 512 (on the higher side) across all workloads, we get KV cache size as 

$$
2 * (512) * (32) * (64) * (22) * 2 = 92MB
$$

- We consider fp 16 even with quantised model because this Jetson module supports quantisation till FP16 only (not even bf16): [](https://forums.developer.nvidia.com/t/quantization-in-tensorrt/201227)(“[Quantization in TensorRt](https://forums.developer.nvidia.com/t/quantization-in-tensorrt/201227)”)
1. **Activation Outputs**: The outputs of each transformer layer (activations) are temporarily stored in GPU memory as the model processes the input tokens. Longer or more complex queries might result in larger or more numerous activation outputs, leading to higher GPU memory usage.
2. **Attention Mechanism**: The attention mechanism requires storing the results of attention scores, weighted sums, and other operations in GPU memory during inference. For longer and more complex queries, the size of the attention buffers will be larger.

[*Same conversational query executed with increasing number of expected output tokens.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfXEdAppnpxPrj0LZYI4pnx_YiH_934tGH4mLHDCbJk3BfJt01hj4r-CwQt0D5tgJhsAVS6acFSA8cSxsXcDFgQ6vvemw9hGpIv0IZfLG_cEqxuibOg-gfR3ZMeAihNiCDGwb5g_gcbnyOVPMXN9z0i_nzx?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Same conversational query executed with increasing number of expected output tokens.*

### GPU Memory Usage and Output Length Analysis

To investigate how GPU memory usage changes with increasing output length, we executed a conversational query with a prompt length equal to the median prompt length for the conversational workload category. We selected a conversational query type because TinyLlama is fine-tuned for chat-related tasks. After excluding the warm-up phase, we repeatedly executed the same query, instructing the model to generate progressively longer responses with each run.

Our expectation was that GPU memory usage would increase proportionally with the length of the output. However, an interesting observation emerged: the GPU memory usage remained constant across all five runs, regardless of the output length. Instead, the time spent by the GPU executing the query increased linearly with the increasing output length.

This unexpected result led us to conduct a reverse experiment to further explore the relationship between input and output tokens. We aimed to observe how GPU memory usage would change when varying the number of input tokens while keeping the number of output tokens constant.

[*(Zoomed version) Four task-oriented queries with different prompt lengths executed with the same number of expected output tokens (50).*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcOsjW_RN0UlbC3YSxpAhWUBJVH4ywc4tiFiNyMhCFn2ne3Db5vf5v3xxP3rFJ6bvqHuJm4L42UOgzOFmq8oRJ_iw9qAv1Jo_5bX4I8LmEL4aT-0YIlXgxyyi_dwADXybJcJbeale5LozNbz6r6mODkA60?key=7VKbPIlbOp2wxPlxUTA4Ug)

*(Zoomed version) Four task-oriented queries with different prompt lengths executed with the same number of expected output tokens (50).*

### GPU Memory Usage with Varying Input Tokens

For this experiment, we selected four different queries from the task-oriented workload. Task-oriented queries were chosen over conversational queries due to the greater variation in the number of input tokens available in this workload. The output length was set to 50 tokens for all four queries, and any data from warm-up runs was discarded.

The zoomed-in plot reveals that GPU memory usage does indeed increase as the number of tokens in the prompt increases. While the change in the number of tokens is not orders of magnitude different, there is still a noticeable difference in GPU memory usage, measured in megabytes (MB).

However, the reasoning behind the lack of observed difference in GPU memory usage for varying output tokens in the previous experiment remains unclear. It is possible that the nature of the query being "conversational" may have influenced this outcome.

[*Total query time comparison for similarly sized contextual and task-oriented query.*](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcTzTZiOSvqUz8t83sYJDBk8CjeKy20cZ-LaI_KxmI2iLDJjLtLfaBmfvVvBING5Ub7UqKRlEmfR-ualLDq-0MROmaudV1Gg7LohCAxPyy55oSvUil_kRlak_bmJosiru0aGghdQE8sozGxxvVQUMDlqJ9R?key=7VKbPIlbOp2wxPlxUTA4Ug)

*Total query time comparison for similarly sized contextual and task-oriented query.*

### **Validation of Query Execution Time Relative to Token Count and Query Complexity**

To validate our hypothesis that both the number of tokens in the prompt and the complexity of the query influence the total time required for execution, we compared contextual queries with task-oriented queries. Specifically, we anticipated that contextual queries, which generally involve longer prompts and greater complexity, would take longer to execute than task-oriented queries.

In our workload plot (total time per query vs. total tokens in prompt), we observed that task-oriented queries generally exhibited higher execution times compared to contextual queries, despite the latter typically having longer prompts. This discrepancy was attributed to the difference in the number of output tokens to be generated: 100 tokens for task-oriented queries versus 50 tokens for contextual queries.

To test this further, we selected one query from each workload with similar prompt lengths and set the number of expected output tokens to 50 for both. After discarding warm-up runs, we found that the contextual query indeed took longer to execute than the task-oriented query when the number of expected output tokens was controlled. This finding confirms that, when output token length is held constant, the complexity and context length of the query significantly impact the total execution time, with contextual queries requiring more time than task-oriented ones.

# Additional Useful Resources

Throughout this project, various papers, blogs, and repositories have been referenced inline within the blog. This section highlights some particularly valuable resources for anyone looking to delve deeper into the topics discussed:

- **Illustrated Transformer** by Jay Alammar: [Link](https://jalammar.github.io/illustrated-transformer/)
- **Understanding Transformers** - YouTube video by Yannic Kilcher: [Link](https://www.youtube.com/watch?v=bQ5BoolX9Ag)
- **Transformer Architecture** - YouTube video by AI Coffee Break with Letitia: [Link](https://www.youtube.com/watch?v=hMs8VNRy5Ys)
- **How Transformers Work** - YouTube video by Valerio Velardo - The Sound of AI: [Link](https://www.youtube.com/watch?v=zxQyTK8quyY)
- **Understanding Transformers and Attention** - YouTube video by Tech With Tim: [Link](https://www.youtube.com/watch?v=IGu7ivuy1Ag)
- **LLM Inference Serving: Survey of Recent Advances and Opportunities**: [Link](https://arxiv.org/pdf/2407.12391)
- **A Survey on Efficient Inference for Large Language Models**: [Link](https://arxiv.org/pdf/2404.14294)
- **LLM Inference Unveiled: Survey and Roofline Model Insights**: [Link](https://arxiv.org/pdf/2402.16363)
- **Super Tiny Language Models**: [Link](https://arxiv.org/pdf/2405.14159)
- **Speeding Up Deep Learning Inference Using TensorRT**: [Link](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/)
- **Jetson Containers by Dusty-Nv**: [Link](https://github.com/dusty-nv/jetson-containers/blob/master/README.md)
- **Awesome Mobile LLMs** - Curated by Steve Laskaridis: [Link](https://github.com/stevelaskaridis/awesome-mobile-llm)
- **LLaMA 2 from the Ground Up** - Blog by Cameron Wolfe: [Link](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)
- **LLM Transformer Inference Guide** - Blog by Baseten: [Link](https://www.baseten.co/blog/llm-transformer-inference-guide/)
- **Mastering LLM Techniques: Inference Optimization** - Blog by Nvidia: [Link](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- **Illustrated GPT-2** by Jay Alammar: [Link](https://jalammar.github.io/illustrated-gpt2/)