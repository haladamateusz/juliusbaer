import pandas as pd
import numpy as np
import pickle
import glob
from fuzzywuzzy import fuzz
import fuzzy
import unidecode
import datetime

from transformers import pipeline

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

entities_set = ["what is my name?", "when is my birthday?", "what is my marital status?", "what is my account number?", "what is my tax residency?", "what is my net worth in millions?", "what is my profession?", "what is my social security number?", "who is my relationship manager?", "what is my highest previous education?"]

file_name = "data/client_features.csv"

file_csv = pd.read_csv(file_name)
names = file_csv["name"].values
names_easier = {name:name.lstrip().rstrip().replace(" ","").replace(",","").replace(".","") for name in names}
reverse_names = {value:key for key, value in names_easier.items()}

all_files = glob.glob("data/audio_data/all/*.wav")

all_pickles = glob.glob("data/pickles/*.pkl")

def metaphone_key(name):
    dmeta = fuzzy.DMetaphone()
    ascii_name = unidecode.unidecode(name)  # Convert to closest ASCII representation
    key = dmeta(ascii_name)
    return key[0] if key else ""

def find_closest_match(target_name, name_list = names):
    target_name = metaphone_key(target_name)
    closest_name = None
    for name in name_list:
        try:
            name_ = metaphone_key(name)
            closest_name = name if closest_name is None else closest_name
            if name_ == target_name:
                closest_name = name
                return name
        except:
            return name
    return closest_name

def get_info(pickle_name):
    
    with open(pickle_name, "rb") as f:
        data = pickle.load(f)
    name = data["what is my name?"]["answer"]
    name = name.lstrip().rstrip()
    name = name.replace(" ", "")
    name = name.replace(",", "")
    name = name.replace(".", "")
    closest_name = find_closest_match(name)
    row_info = file_csv[file_csv["name"] == closest_name]

    empty_answer = data["what is my name?"]["answer"]

    for k, v in data.items():
        if v["answer"] == empty_answer:
            v["score"] = 0

    for entity in entities_set:
        if entity =="what is my name?":
            continue

        if data[entity]["score"] > 0.3:
            if entity == "when is my birthday?":
                answer = data[entity]["answer"]
                actual_answer = row_info["birthday"].values[0]
                dd, mm, yyyy = actual_answer.split(".")
                if dd not in answer:
                    return False
                if yyyy not in answer:
                    return False
            if entity == "what is my marital status?":
                answer = data[entity]["answer"]
                actual_answer = row_info["marital_status"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "what is my account number?":
                answer = data[entity]["answer"]
                actual_answer = row_info["account_nr"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "what is my tax residency?":
                answer = data[entity]["answer"]
                actual_answer = row_info["tax_residency"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "what is my net worth in millions?":
                answer = data[entity]["answer"]
                actual_answer = row_info["net_worth_in_millions"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "what is my profession?":
                answer = data[entity]["answer"]
                actual_answer = row_info["profession"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "what is my social security number?":
                answer = data[entity]["answer"]
                actual_answer = row_info["social_security_number"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "who is my relationship manager?":
                answer = data[entity]["answer"]
                actual_answer = row_info["relationship_manager"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
            if entity == "what is my highest previous education?":
                answer = data[entity]["answer"]
                actual_answer = row_info["highest_previous_education"].values[0]
                if fuzz.partial_token_set_ratio(answer, actual_answer) < 90:
                    return False
    return True

if __name__ == "__main__":
    L = []
    for file in all_pickles:
        value = get_info(file)
        L.append((value, file))
    with open("data/answers_task2.txt", "w") as f:
        for value, file in L:
            f.write(f"{file} {value}\n")