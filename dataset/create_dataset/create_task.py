import re
import random
from datasets import load_dataset

dataset = load_dataset('Salesforce/dialogstudio', 'MULTIWOZ2_2')

data = set()
while len(data) < 100:
    sample = dataset["test"][random.randint(0, len(dataset["test"])-1)]
    prompt = sample["prompt"][0].replace(" and external database", "")
    task = sample["log"][-1]["dialog history"] # pick the last dialog to get whole history
    task = re.sub("<SYSTEM>((.|\n)(?!<USER>))+[ ?.]", "", task) # remove system responses from the history
    task = re.sub("[0-9A-Z]{8,}", "", task)
    task_sample = prompt + " : " + task + "\n"
    if len(task_sample) < 1600:
        data.add(task_sample)
    
with open(r"C:\Users\aayus\Downloads\queries\workload-queries\task-oriented.txt", "w") as output:
    output.write("".join(list(data)))