import os
from transformers import pipeline

ner_pipeline = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")

entities_set = ["what is my name?","when is my birthday?", "what is my marital status?", "what is my account number?", "what is my tax residency?", "what is my net worth in millions?", "what is my profession?", "what is my social security number?", "who is my relationship manager?", "what is my highest previous education?"]

def obtain_info(file_name):
    diction = {}
    with open(file_name, "r") as f:
        text = f.read()
    for entity in entities_set:
        result = ner_pipeline(question=entity, context=text)
        diction[entity] = {"score": result["score"], "answer": result["answer"]}
    return diction

for file_name in os.listdir("data/transcriptions/all"):
    diction_overall = {}
    diction = obtain_info(f"data/transcriptions/all/{file_name}")
    diction_overall[file_name] = diction
    
