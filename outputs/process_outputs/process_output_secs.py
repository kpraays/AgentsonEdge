import os
import csv

def get_data(data_file):
    no_output = False
    previous_row = None        
    with open(data_file, "r", encoding="utf-8") as data:
        cpu_percent = []
        cpu_mem = []
        gpu_mem = []
        gpu_per = []
        print(data_file)
        for row in data:
            if previous_row is not None:
                if "'main']" in row:
                    vals = row.split("main")[0].split(",")[-4:-1]
                    # print(vals)
                    cpu_percent.append(float(vals[0].strip()))
                    cpu_mem.append(float(vals[1].strip()) * 4/1024)
                    gpu_mem.append(float(vals[2].strip()) * 4/1024)
                
                    # only get gpu data when others data points are present
                    if "'GPU': " in previous_row:
                        gpu = float(previous_row.split("'GPU': ")[1].split(",")[0])
                        gpu_per.append(gpu)
            
            previous_row=row                                
            
            if "llama_print_timings:        eval time" in row:
                tpt = float(row.split(" ")[-4])
            
            if "llama_print_timings: prompt eval time" in row:
                petpt = float(row.split(" ")[-4])
                
            if "prompt is too long" in row:
                no_output = True
                
    if no_output:
        return [-1,-1], [-1,-1], [-1,-1], -1, -1, [-1,-1]
    return cpu_percent, cpu_mem, gpu_mem, tpt, petpt, gpu_per

def get_max(float_list):
    max = -1
    for i in float_list:
        if i > max:
            max = i
    return max

def get_average(float_list):
    sum = 0
    for i in float_list:
        sum += i
    return sum/ len(float_list)

# print(cpu_percent[1:])
# print(cpu_mem[1:])
# print(gpu_mem[1:])
# print(tpt)
    
if __name__ == "__main__":
    
    src_direc = r"C:\Users\aayus\Downloads\output_plots\output_readings\workload_output_100_jetson"
    dest_direc = r"C:\Users\aayus\Downloads\output_plots\output_readings"
    for path, dirs, files in os.walk(src_direc):
        data_files = [os.path.join(path, file) for file in files]
        data_files = sorted(data_files)
    count = 0
    with open(os.path.join(dest_direc, "processed_secs.csv"), "w", newline="") as csvfile:
        fieldnames = ["secs", "id", "cpu_per", "cpu_mem", "gpu_mem", "gpu_per",  "generation_throughput", "prompt_eval_throughput"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()    
        
        for i, data_file in enumerate(data_files):    
            cpu_percent, cpu_mem, gpu_mem, tpt, petpt, gpu_per = get_data(data_file)
            qid = data_file.split(os.path.sep)[-1].split("_")
            query_id = f"{qid[0]}_{int(qid[1]):02d}"

            # size of recorded arrays is the same
            for index, value in enumerate(cpu_percent):
                # per second record
                
                if cpu_percent[index]!=-1:
                    writer.writerow({"secs": count,
                                    "id": query_id, 
                                    "cpu_per": cpu_percent[index],
                                    "cpu_mem": cpu_mem[index],
                                    "gpu_mem": gpu_mem[index],
                                    "gpu_per": gpu_per[index],
                                    "generation_throughput": tpt, 
                                    "prompt_eval_throughput": petpt})
                    count += 1                    
        



    


        
        
