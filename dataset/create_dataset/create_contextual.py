import json
import random

with open(r"C:\Users\aayus\Downloads\queries\QuAC\val_v0.2.json", "r") as data:
    content = json.load(data)
    
    
# # Explore the dataset    
# topics = []    
# for item in content["data"]:
#     topics.append(item["title"])
#     if item["title"] == "Force":
#         for element in item['paragraphs'][0]["qas"]:
#             print(element)
#             print()


# print(topics)
# ['Normans', 'Computational_complexity_theory', 'Southern_California', 'Sky_(United_Kingdom)', 'Victoria_(Australia)', 'Huguenot', 'Steam_engine', 'Oxygen', '1973_oil_crisis', 'European_Union_law', 'Amazon_rainforest', 'Ctenophora', 'Fresno,_California', 'Packet_switching', 'Black_Death', 'Geology', 'Pharmacy', 'Civil_disobedience', 'Construction', 'Private_school', 'Harvard_University', 'Jacksonville,_Florida', 'Economic_inequality', 'University_of_Chicago', 'Yuan_dynasty', 'Immune_system', 
# 'Intergovernmental_Panel_on_Climate_Change', 'Prime_number', 'Rhine', 'Scottish_Parliament', 'Islamism', 'Imperialism', 'Warsaw', 'French_and_Indian_War', 'Force']

# get 100 questions - extract one questions from a random topic

def get_random_sample(data):
    return data[random.randint(0, len(data)-1)]

q_ids = set()
question_list = []
while len(q_ids) < 100:
    data = get_random_sample(content["data"])
    # background = data["background"]
    # words = background.split()
    questions = get_random_sample(data["paragraphs"])
    background = questions["context"]
    questions = questions["qas"]
    q = get_random_sample(questions)
    if "question" in q and q["id"] not in q_ids and len(background) < 1600:
        q_ids.add(q["id"])
        question_list.append("::CONTEXT:: " + background.replace(" CANNOTANSWER", "") + " ::question:: " + q["question"]+" ")
 
with open(r"C:\Users\aayus\Downloads\queries\workload-queries\contextual.txt", "w") as output:
    data = "\n".join(question_list)
    output.write(data.lower())
    