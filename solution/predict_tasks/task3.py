import pandas as pd

client_info = pd.read_csv("data/client_features.csv")

genders = client_info[["name"," gender"]]
genders.set_index("name", inplace=True)

with open("predict_tasks/name_to_file.txt") as f:
    name_to_file = f.readlines()

name_to_file = [x.rstrip("\n") for x in name_to_file]

name_to_file = {x.split(".pkl")[0].replace("data/pickles/","")+".wav":x.split(".pkl")[1].lstrip().rstrip() for x in name_to_file}

name_to_file = pd.DataFrame.from_dict(name_to_file, orient="index", columns=["name"])


with open("predict_tasks/gender_classific.txt", "r") as f:
    gender_predicted = f.readlines()

gender_predicted = {x.split(" ")[1].rstrip("\n"):x.split(" ")[0] for x in gender_predicted}

gender_predicted = pd.DataFrame.from_dict(gender_predicted, orient="index", columns=["pred"])

name_to_file["gender"] = [genders.loc[x].values[0] for x in name_to_file["name"]]
final_df = pd.concat([name_to_file, gender_predicted], axis=1)
final_df["pred"] = final_df["pred"].map(lambda x: "M" if "male" == x else "F")
final_df["corrected"] = final_df.apply(lambda x: 1 if x["pred"] in x["gender"] else 0, axis=1)
final_df.drop(columns=["gender","name", "pred","correct"], inplace=True)
final_df.to_csv("third_task.csv", )