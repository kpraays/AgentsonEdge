# for workload level stuff

import os
import csv

def get_data(data_file):
    no_output = False
   
    with open(data_file, "r", encoding="utf-8") as data:
        cpu_percent = []
        print(data_file)
        for row in data:
            if "'main']" in row:
                vals = row.split("main")[0].split(",")[-4:-1]
                # print(vals)
                cpu_percent.append(float(vals[0].strip()))
                                           
            if "llama_print_timings:       total time" in row:
                total_time = float(row.split("/")[0].strip().split(" ")[-2])
                total_tokens = float(row.split("/")[1].strip().split(" ")[0])
                
            if "llama_print_timings:        eval time" in row:
                TBT = float(row.split("(")[1].strip().split(" ")[0])
            
            if "llama_print_timings: prompt eval time" in row:
                TTFT = float(row.split("/")[0].strip().split(" ")[-2])
                prompt_tokens = float(row.split("/")[1].strip().split(" ")[0])
                
            if "prompt is too long" in row:
                no_output = True
                
    if no_output:
        return -1, -1, -1, -1, -1, [-1, -1]
    return TTFT, TBT, total_time, total_tokens, prompt_tokens, cpu_percent

    
if __name__ == "__main__":
    
    src_direc = r"C:\Users\aayus\Downloads\output_plots\output_readings\granular_helper\contextual_task-oriented_same_output_length_total_time\measurements"
    dest_direc = r"C:\Users\aayus\Downloads\output_plots\output_readings\granular_helper\contextual_task-oriented_same_output_length_total_time"
    for path, dirs, files in os.walk(src_direc):
        data_files = [os.path.join(path, file) for file in files]
        data_files = sorted(data_files)
    count = 0
    
    with open(os.path.join(dest_direc, "processed_latency_same_output_compare.csv"), "w", newline="") as csvfile:
        fieldnames = ["secs", "id", "TTFT", "TBT", "total_time", "total_tokens", "prompt_tokens"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()    
        
        for i, data_file in enumerate(data_files):    
            TTFT, TBT, total_time, total_tokens, prompt_tokens, cpu_percent = get_data(data_file)
            qid = data_file.split(os.path.sep)[-1].split("_")
            query_id = f"{qid[0]}_{int(qid[1]):02d}"

            # size of recorded arrays is the same
            for index, value in enumerate(cpu_percent):
                # per second record
                
                if cpu_percent[index]!=-1:
                    writer.writerow({"secs": count,
                                    "id": query_id, 
                                    "TTFT": TTFT,
                                    "TBT": TBT,
                                    "total_time": total_time,
                                    "total_tokens": total_tokens,
                                    "prompt_tokens": prompt_tokens})   
                    count += 1    
