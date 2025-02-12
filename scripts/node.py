import threading
import asyncio
import os
import sys
from utils_simulation import get_hospitals, load_dataset, print_line, set_reproducibility
from utils_collaborator import *
from utils_manager import *
from brownie import FederatedLearning, network, accounts
import ipfshttpclient
import json
import time
import numpy as np
import random
import tensorflow as tf
import logging
from deploy_FL import get_account
from tensorflow.keras.models import model_from_json


ROLE = 'collaborator'
hospital_name = sys.argv[3]

if hospital_name == 'aggregator':
    raise ValueError("Select a suitable name for the node. \"aggregator\" might create conflicts with other parameters.")


if 'aggregator' in sys.argv:
    ROLE = 'aggregator'


# Configurazione IPFS e Blockchain
IPFS_client = ipfshttpclient.connect()
FL_contract = FederatedLearning[-1]

# Dataset condiviso
hospitals = get_hospitals()
hospital_dataset = load_dataset(hospitals)
test_dataset = hospital_dataset['test']
manager = get_account()  # Indirizzo del manager (solo se ROLE è "aggregator")

NUM_ROUNDS = 5

class Collaborator(threading.Thread):
    def __init__(self, hospital_name):
        super().__init__()
        self.hospital_name = hospital_name
        self.hospital = hospitals[hospital_name]

    def retrieve_model(self):
        retrieve_model_tx = FL_contract.retrieve_model({"from": self.hospital.address})
        retrieve_model_tx.wait(1)
        model_data = decode_utf8(retrieve_model_tx)
        custom_objects = {'FedAvg': FedAvg, 'FedProx': FedProx}
        self.hospital.model = model_from_json(model_data, custom_objects=custom_objects)

    def train_local_model(self):
        train_dataset = hospital_dataset[self.hospital_name]
        epochs = random.randint(1, NUM_EPOCHS)
        for epoch in range(epochs):
            for imgs, labels in train_dataset:
                self.hospital.model.train_step(imgs, labels)
        self.hospital.weights = self.hospital.model.get_weights()

    def send_weights(self):
        weights_bytes = weights_encoding(self.hospital.weights)
        add_info = IPFS_client.add(weights_bytes, pin=True)
        hash_encoded = add_info["Hash"].encode("utf-8")
        send_weights_tx = FL_contract.send_weights(hash_encoded, {"from": self.hospital.address})
        send_weights_tx.wait(1)

    def run(self):
        print(f"[Collaborator] Nodo {self.hospital_name} avviato")
        self.retrieve_model()
        for round_num in range(NUM_ROUNDS):
            print(f"[Collaborator] Round {round_num + 1}: Training in corso...")
            self.train_local_model()
            self.send_weights()

class Aggregator(threading.Thread):
    def __init__(self):
        super().__init__()
        self.global_weights = None

    def retrieve_weights(self):
        hospitals_addresses = FL_contract.get_collaborators({"from": manager})
        hospitals_weights = {}
        for address in hospitals_addresses:
            weights_hash = FL_contract.retrieve_weights(address, {"from": manager}).decode("utf-8")
            weights_encoded = IPFS_client.cat(weights_hash)
            hospitals_weights[address] = weights_decoding(weights_encoded)
        return hospitals_weights

    def aggregate_weights(self, hospitals_weights):
        weights_dim = len(next(iter(hospitals_weights.values())))
        aggregated_weights = []
        for i in range(weights_dim):
            layer_weights = [weights[i] for weights in hospitals_weights.values()]
            aggregated_weights.append(sum(layer_weights) / len(hospitals_weights))
        return aggregated_weights

    def distribute_weights(self, aggregated_weights):
        aggregated_weights_bytes = weights_encoding(aggregated_weights)
        res = IPFS_client.add(aggregated_weights_bytes, pin=True)
        hash_encoded = res["Hash"].encode("utf-8")
        send_aggregated_weights_tx = FL_contract.send_aggregated_weights(hash_encoded, {"from": manager})
        send_aggregated_weights_tx.wait(1)

    def run(self):
        print("[Aggregator] Nodo aggregatore avviato")
        for round_num in range(NUM_ROUNDS):
            print(f"[Aggregator] Round {round_num + 1}: Aggregazione in corso...")
            hospitals_weights = self.retrieve_weights()
            self.global_weights = self.aggregate_weights(hospitals_weights)
            self.distribute_weights(self.global_weights)


if __name__  == 'main':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Main  : created node entity for " + hospital_name + "\nRole: " + ROLE)
    
    if ROLE == 'collaborator':
        thread = Aggregator()
    elif ROLE == 'aggregator':
        thread = Collaborator(hospital_name=hospital_name)
    else:
        raise ValueError(f"{ROLE} is an unknown role. Possible roles: \"aggregator\" or \"collaborator\"")

