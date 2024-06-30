import os
import sys
from modelling.generate_transcripts import generate_transcripts_from_audio
from representations.Client import Client
import pandas as pd
import string
import pickle
from fuzzywuzzy import process, fuzz
class BankClient:
    def __init__(self, file_path="data/client_features.csv", bank_id=None):
        '''
            Initialize the BankClient object
            This takes a csv file as input and populates the clients dictionary.
            INPUTS:
            file_path: str: Path to the csv file containing the client information
            bank_id: str: Unique identifier for the bank
        '''
        if bank_id is None:
            bank_id = 42 ## Default bank_id
        self.bank_id = bank_id
        self.clients = {}
        self.file_path = file_path
        self.populate_clients()

    def populate_clients(self, transcripts_extracted=True):
        '''
            Populate the clients dictionary with the client information from the csv file
            INPUTS:
            file_path: str: Path to the csv file containing the client information
        '''
        ##TODO: Ensure that the clients have already not been populated. If they have been, return
        client_info_overall = pd.read_csv(self.file_path)
        client_names = client_info_overall['name']
        for name in client_names:
            if f"{name}.pkl" not in os.listdir("data/cached_clients/"):
                print(name)
                print(os.listdir("data/cached_clients/"))
                self.clients[f"client_{name}"] = Client(name)
                client_info = client_info_overall.loc[client_info_overall['name'] == name].to_dict()
                client_info = {k: list(v.values())[0] for k,v in client_info.items()}
                client_info.pop('name')
                self.clients[f"client_{name}"].populate_client_info(client_info)
                # self.clients[f"client_{name}"].save_self()
            else:
                self.clients[f"client_{name}"] = self.load_client(name)
                print(f"Loaded client {name}")
        if transcripts_extracted:
            if os.path.exists("data/transcriptions/all"):
                text_files = os.listdir("data/transcriptions/all")
                for text_file in text_files:
                    with open(f"data/transcriptions/all/{text_file}", "r") as f:
                        text_file_content = f.read()
                    name_client = process.extractOne(text_file_content, client_names.to_list(), scorer=fuzz.partial_token_set_ratio)
                    self.clients["client_" + name_client[0]].transcript.append( text_file_content)
                    self.clients["client_" + name_client[0]].audio_files.append(text_file.replace(".txt", ""))
                    self.clients["client_" + name_client[0]].cache_self()
    def load_client(self, name):
        '''
            Load the client from the cached file
            INPUTS:
            name: str: Name of the client
            OUTPUT:
            client: Client: The client object
        '''
        with open(f"data/cached_clients/{name}.pkl", "rb") as f:
            client = pickle.load(f)
        return client

    def get_embedding(self, audio_file_name):
        '''
            Get the embedding for the audio file
            INPUTS:
            audio_file_name: str: Path to the audio file
            OUTPUT:
            embedding: np.array: The embedding for the audio file
        '''
        pass

    def get_closest_client(self, embedding):
        '''
            Get the closest client to the embedding. Right now as first step, we will be using cosine similarity to obtain a ranking of the clients
            INPUTS:
            embedding: np.array: The embedding for the audio file
            OUTPUT:
            closest_client: Client: The closest client to the embedding
        '''
        pass

if __name__ == "__main__":
    bank_client = BankClient()
