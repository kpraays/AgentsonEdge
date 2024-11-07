import pandas as pd

splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])

import random

questions = df["question"].unique()   

pos = set()
while len(pos) < 100:
    pos.add(random.randint(0, len(questions)))
    
data = ""    
for index in pos:
    data += questions[index] + "\n"
    
with open(r"C:\Users\aayus\Downloads\queries\workload-queries\simple.txt", "w") as output:
    output.write(data)
    
    