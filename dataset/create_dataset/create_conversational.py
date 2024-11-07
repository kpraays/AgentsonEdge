import pandas as pd
import re
import random


df = pd.read_csv("hf://datasets/michellejieli/friends_dataset/friends_cleaned.csv")

dialogues = df["text"].unique()

filtered = set()
for d in dialogues:
    if len(d) > 80 and len(d)<150:
        filtered.add(re.sub("[^0-9a-zA-Z ,.?!-]", "", d))

data = []
while len(data) < 100:
    filter_list = list(filtered)
    element = filter_list[random.randint(0, len(filter_list)-1)]
    filtered.remove(element)
    data.append(element)

with open(r"C:\Users\aayus\Downloads\queries\workload-queries\conversational.txt", "w") as output:
    output.write("\n".join(data))
