import os
import sys
from modelling.generate_transcripts import generate_audio
import pickle

class Client:
    def __init__(self, client_name, client_info=None, embeddings=None):
        '''
            Populate the client information
        '''
        self.client_name = client_name
        self.client_info = client_info  # Information about the client
        self.audio_files = []
        self.embeddings = []  # List to store embeddings
        self.transcript = []

    def cache_self(self):
        with open(f"data/cached_clients/{self.client_name}.pkl", "wb") as f:
            pickle.dump(self, f)

    def populate_client_info(self, info):
        ## From the file_path, parse the different attributes of the client and populate the client_info
        self.client_info = info

    def generate_transcripts_from_audio_file(self, audio_file_name):
        ### TODO: Read the transcript from the audio file. First check if the transcript is already present in the database; else generate it
        pass

    def generate_embeddings_from_audio_file(self, audio_file_name):        
        ### TODO: Read the transcript from the audio file. Check if the embedding already exists. Else generate it
        pass

    def generate_embedding(self, audio_file):
        # Logic to generate embedding from audio_file
        new_embedding = self.generate_embedding_from_audio(audio_file)
        return new_embedding
    
    def __str__(self):
        return f"Client ID: {self.client_id}, Client Info: {self.client_info}"
    
    def characteristics_match(self, ):
        pass