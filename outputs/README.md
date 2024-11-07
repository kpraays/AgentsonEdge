## Experiments



1. **Category-Wise Query Submission**:
    * **Process**: We submitted 100 queries from each category sequentially. Every 0.5 seconds, the following metrics were recorded:
        1. **CPU and GPU Memory Usage, CPU Utilization**: These metrics were recorded (across all four cores: CPU utilization) for the query process executed through llama.cpp for TinyLlama.
        2. **System-Wide GPU Utilization**: GPU utilization was monitored while the query execution process was in progress. During data processing, the GPU utilization values used for plots were those that coincided with the same time periods for which per-process readings existed.
        3. **Per Query Metrics**:
            1. **Time to First Token**: The duration before the first token appeared.
            2. **Time Between Tokens**: The time gap observed between the generation of each successive token after the first token had appeared. \

2. **Granular Query Analysis**:
    * **Process**: We calculated the median length of prompt tokens for each query category and selected one representative query from each category that closely matched the median token length. For each representative query, the same measurements as above were recorded every centisecond (1 centisecond equals 1/100th of a second, corresponding to 100 ticks per second on Ubuntu systems). Recordings were made after a warm-up phase, as discrepancies were found in the CPU usage of complex queries when measurements were taken without warming up. The initial four readings for other queries in each workload were discarded to ensure consistency.

**Additional Granular Measurements for Special Scenarios**:

For each of the following scenarios, a warm-up phase was included to ensure consistency in measurements.



    1. **Task-Oriented Queries**:
        * **Process**: GPU memory usage was recorded every centisecond for four task-oriented queries. These queries were selected based on an increasing number of tokens in the prompt, with the output token count fixed at 50. \

    2. **Total Time Measurement**:
        * **Process**: The total time taken to complete two queries was recorded—one from the task-oriented workload and one from the contextual workload. These queries were selected to minimize the difference in the number of tokens in the prompt between them. \

    3. **Conversational Workload Query**:
        * **Process**: A single conversational query with a number of prompt tokens equal to the category median was selected. GPU memory usage was recorded over five runs of the same query, with the expected output length increasing incrementally in each run. The output token counts were set to ["10", "50", "120", "230", "400"].



## Measuring System Resource Usage During Inference**

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
