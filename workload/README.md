## What was done?


**Measuring System Resource Usage During Inference**


    Our measuring script loaded all queries in the dataset, appended an instruction based on the category, and invoked the *llama.cpp* command to execute each query (the appended instruction per category can be found in the appendix). The output was recorded in a text file identified by a unique ID, comprising the query category and number. To ensure feasible run durations and prevent overheating the board, we imposed limits on output length for each category.


    For each query submission, a new process was spawned, and the associated resource utilization was recorded for that process ID. The results of query execution were captured by connecting to the process's output and error pipes.


    **Tools Used for Measuring Resource Utilization**:



* **Psutil**:
    * cpu_percent: Recorded the CPU utilization of the process as a percentage.
    * memory_percent: Measured the process memory usage as a percentage of the total physical system memory. \

* **Jetson Stats**:
    * jetson.stats: Provided a simplified version of tegrastats, allowing for easy logging of the NVIDIA Jetson status, including metrics such as CPU, RAM, fan speed, temperature, GPU usage, and uptime. \

    * jetson.processes: Returned a list of all processes running on the GPU, including CPU usage, GPU memory usage, and CPU memory usage (“Jetson-Stats”).

    **Command Format for Query Invocation**: Each query was executed using the following command format:


    /path/to/compiled/main/bin-for-llama.cpp -m /model/gguf-file-location -p “Query text” -n &lt;tokens-expected>
