import subprocess
import psutil
import time
import os
from jtop import jtop 

query_category_prompts = {"complex": "You are a helpful assistant. Give a short answer. Query: $$**$$ Answer: ", 
                          "contextual": "You are a very helpful assistant. Answer the question: $$**$$ Answer: ",
                          "conversational": "You are a helpful assistant.> User: $$**$$? Assistant: ",
                          "simple": "You are a helpful assistant who knows everything. Give a short answer. Query: $$**$$ Answer: ",
                          "task-oriented": "$$**$$ <BOT>: Great! Based on the provided information, following are the steps: "}


def load_queries(file_name):
    queries = []
    with open(file_name, "r") as query_file:
        for row in query_file:
            if len(row) > 1:
                queries.append(row)
    
    return queries

def load_files(location):
    data_files = []
    for path, directories, files in os.walk(location):
        for file in files:
            print('found %s' % os.path.join(path, file))
            data_files.append(os.path.join(path, file))
    return data_files

def create_query_dict(data_files):
    query_dict = {}
    for file in data_files:
        queries = load_queries(file)
        category = file.split(os.path.sep)[-1].split(".")[0]
        query_dict[category] = queries
    
    return query_dict

def process_queries(query_dict, query_category_prompts):
    query_set = {}
    for key in query_dict.keys():
        query_ids = {}
        append_prompt = query_category_prompts[key]
        for index, query in enumerate(query_dict[key]):
            # query_dict[key] contains a list of queries for that category.
            query_ids[key+"_"+str(index)] = append_prompt.replace("$$**$$", query.replace("\n", ""))
        query_set[key] = query_ids
        
    # if "contextual" in query_set:
    #     context_queries = query_set["contextual"]
    #     context = {}
    #     position = 0
    #     for index in range(0, len(context_queries), 2):
    #         context["contextual_"+str(position)] = (context_queries["contextual_"+str(index)], context_queries["contextual_"+str(index+1)])
    #         position += 1
    #     query_set["contextual"] = context
    
    return query_set

### test query set creation methods
# data_files = load_files("/home/nanojet/nano-llama/workload/example-queries")
# query_dict = create_query_dict(data_files)
# query_set = process_queries(query_dict, query_category_prompts)
# for key, value in query_set.items():
#     print(f"{key} --> {value}")

def execute_query(args):
    output = ""
    jetson = jtop()
    jetson.start()
    time.sleep(1)
    stat = jetson.stats
    output = output + str(stat) + "\n"
    
    # Run the model query and measure time
    start_time = time.time()
    popen = subprocess.Popen(args , stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output = output + f"pid: {popen.pid}" + "\n"
    
    time.sleep(0.2)
    process = psutil.Process(popen.pid)
    while popen.poll() is None:
        output = output + f"psutil memory: {process.memory_percent()}" + "\n"
        output = output + f"psutil cpu percent: {process.cpu_percent()}" + "\n"
        time.sleep(0.01) # because unix ticks per second = 100
        output = output + str(jetson.stats) + "\n"
        output = output + str(jetson.processes) + "\n"

    (results, errors) = popen.communicate()
    end_time = time.time()
    
    # Calculate execution time
    execution_time = (end_time - start_time) * 1000  # in milliseconds
    
    jetson.close()    
    output = output + str(results) + "\n"
    output = output + str(errors) + "\n"
    output = output + f"Time taken is: {execution_time}" + "\n"
    
    return output

def write_output(output, query_id, timestamp):
    destination = os.path.join("output", timestamp)
    os.makedirs(destination, exist_ok=True)
    with open(os.path.join(destination, query_id), "w") as data_out:
        data_out.write(output)

def execute_workload(query_set, tokens):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%m-%S")
    count=0
    for key, q_dict in query_set.items():
        for query_id, query in q_dict.items():
            args = ['/home/nanojet/nano-llama/llama.cpp/main', '-m', '/home/nanojet/nano-llama/llama.cpp/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf', '-p']
            args.append(f'"{query}"')
            args.append("-n")
            args.append(tokens[key])
            output = execute_query(args)
            write_output(output, query_id, timestamp)
            print(args)
            count+=1
            


def main():
    # data_files = load_files("/home/nanojet/nano-llama/workload/example-queries")
    data_files = load_files("/home/nanojet/nano-llama/workload/workload-queries")
    query_dict = create_query_dict(data_files)
    query_set = process_queries(query_dict, query_category_prompts)

    # tokens = {"complex": "70", 
    #             "contextual": "50",
    #             "conversational": "40",
    #             "simple": "30",
    #             "task-oriented": "100"}
    tokens = {"complex": "70", 
                "contextual": "50",
                "conversational": "40",
                "simple": "30",
                "task-oriented": "50"}    
    
    
    # query_dict = {
    #     'simple': {
    #         'simple_0': 'You are a helpful assistant who knows everything. Give a short answer for the following query: What is the capital of Japan? Answer:'
    #         }
    #     }
    
    execute_workload(query_set, tokens)
    
    # /home/nanojet/nano-llama/llama.cpp/main -m /home/nanojet/nano-llama/llama.cpp/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf -p You are a helpful AI assistant. Write a short answer for the query: How many continents are there?
    # /home/nanojet/nano-llama/llama.cpp/main -m /home/nanojet/nano-llama/llama.cpp/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf -p "As a helpful AI assistant, answer the following query. Keep the answers to the point and small: Explain the theory of evolution by natural selection."
    # /home/nanojet/nano-llama/llama.cpp/main -m /home/nanojet/nano-llama/llama.cpp/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf -p "You are a very helpful AI assistant who gives to the point responses to the user: Hi, how are you today?"

    # args = ['/home/nanojet/nano-llama/llama.cpp/main', '-m', '/home/nanojet/nano-llama/llama.cpp/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf', '-p', '"You are a helpful AI assistant. Write a short answer for the query: How many continents are there?"']

    
if __name__ == "__main__":
    main()