import os
import sys
import asyncio
import json
import time
import pickle
import random
import numpy as np
import tensorflow as tf
from numpy import require
from tensorflow.keras.models import model_from_json

# Brownie and IPFS imports
from brownie import FederatedLearning
import ipfshttpclient

# Utility functions and constants (adjust the imports as needed)
from utils_simulation import get_hospitals, print_line, set_reproducibility, round_out_of_battery, device_out_of_battery, load_dataset
from utils_collaborator import *  # (import only what you need)
from fedAvg import FedAvg
from fedProx import FedProx
from constants import *

# Suppress TensorFlow logging and set reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_reproducibility()

# Ensure the directory containing this script is in sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)


class Collaborator:
    def __init__(self, hospital_name, out_of_battery=False, network=None):
        # Instead of reading sys.argv directly, use the passed parameter:
        self.hospital_name = hospital_name
        
        # Retrieve hospitals and load dataset
        self.hospitals = get_hospitals()
        self.hospital_dataset = load_dataset(self.hospitals)
        self.test_dataset = self.hospital_dataset['test']

        # Connect to IPFS and the Blockchain contract
        self.IPFS_client = ipfshttpclient.connect()
        self.FL_contract = FederatedLearning[-1]
        self.contract_events = self.FL_contract.events

        # Process command-line arguments
        #if len(sys.argv) < 6:
        #    print("Usage: collaborator_parallel.py hospital_name [out_of_battery] --network network_name")
        #    sys.exit(1)
        #self.hospital_name = sys.argv[3]

        # Initialize evaluation storage for this collaborator
        self.hospitals_evaluation = {self.hospital_name: []}

        # Battery parameters and file naming
        if out_of_battery:
            self.ROUND_BATTERY = round_out_of_battery(NUM_ROUNDS)
            self.DEVICES_OUT_OF_BATTERY = device_out_of_battery(get_hospitals(), n=NUM_DEVICES_OUT_OF_BATTERY)
            self.name = f"{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{self.DEVICES_OUT_OF_BATTERY}"
            print(f"Device/s {self.DEVICES_OUT_OF_BATTERY} will be out of battery at round {self.ROUND_BATTERY + 1}")
        else:
            self.ROUND_BATTERY = 100
            self.DEVICES_OUT_OF_BATTERY = []
            self.name = f"{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_[]"
            print("No devices out of battery")

        # Save the devices out of battery info
        with open('devices_out_of_battery.pkl', 'wb') as file:
            pickle.dump(self.DEVICES_OUT_OF_BATTERY, file)

        # Dictionary for tracking gas fees for this collaborator
        self.gas_fee_collab = {
            self.hospital_name: {
                'retrieve_fee': [],
                'send_fee': [],
                'model_start_fee': 0
            }
        }

    def closeState_alert(self, event):
        print("The FL Blockchain has been CLOSED\n")
        print("RESULTS - Hospitals Performance Evaluation through Federated Learning...")
        for hosp_name in self.hospitals_evaluation:
            print(f"{hosp_name}:")
            for round_idx, (loss, acc) in enumerate(self.hospitals_evaluation[hosp_name], start=1):
                print(f"\tRound {round_idx}:\tLoss: {loss:.3f} - Accuracy: {acc:.3f}")
        # Uncomment if you want to disconnect and exit:
        # sys.exit(0)

    def start_event(self):
        print("Hello hospital " + self.hospital_name + "!!")

        # Retrieve the model provided by the Manager
        retrieve_model_tx = self.FL_contract.retrieve_model(
            {"from": self.hospitals[self.hospital_name].address}
        )
        self.gas_fee_collab[self.hospital_name]['model_start_fee'] = retrieve_model_tx.gas_used
        retrieve_model_tx.wait(1)

        custom_objects = {'FedAvg': FedAvg, 'FedProx': FedProx}
        decoded_model = decode_utf8(retrieve_model_tx)
        model = model_from_json(decoded_model, custom_objects=custom_objects)
        print("Model ", model)
        self.hospitals[self.hospital_name].model = model

        # Retrieve compile information from the Manager
        retreive_compile_info_tx = self.FL_contract.retrieve_compile_info(
            {"from": self.hospitals[self.hospital_name].address}
        )
        self.gas_fee_collab[self.hospital_name]['model_start_fee'] += retreive_compile_info_tx.gas_used
        retreive_compile_info_tx.wait(1)

        decoded_compile_info = decode_utf8(retreive_compile_info_tx)
        fl_compile_info = json.loads(decoded_compile_info)
        self.hospitals[self.hospital_name].compile_info = fl_compile_info

        # Compile the model with the received compile information
        self.hospitals[self.hospital_name].model.compile(**self.hospitals[self.hospital_name].compile_info)

    def round_loop(self, round_idx, fed_dict, file_name):
        if self.hospital_name not in fed_dict:
            fed_dict[self.hospital_name] = {}
        if round_idx >= self.ROUND_BATTERY and self.hospital_name in self.DEVICES_OUT_OF_BATTERY:
            print(f"Device {self.hospital_name} is out of battery")
            fed_dict[self.hospital_name][round_idx] = "out_of_battery"
        else:
            print(f"Device {self.hospital_name} is training ...")
            fed_dict = self.fitting_model_and_loading_weights(self.hospital_name, round_idx, fed_dict)

        path = './results/' + file_name + '.json'
        with open(path, 'w') as json_file:
            json.dump(fed_dict, json_file)
        return fed_dict

    def aggregatedWeightsReady_event(self, round_idx):
        if self.hospital_name in self.DEVICES_OUT_OF_BATTERY and (round_idx + 1) >= self.ROUND_BATTERY:
            return
        print("Retrieving weights for hospital ", self.hospital_name)
        self.retrieving_aggreagted_weights(self.hospital_name)
        print("-" * 50)
        print()

    def fitting_model_and_loading_weights(self, _hospital_name, round_idx, fed_dict):
        train_dataset = self.hospital_dataset[_hospital_name]
        # Randomize the number of epochs if the model is of type FedProx
        if isinstance(self.hospitals[_hospital_name].model, FedProx):
            epochs = random.randint(1, NUM_EPOCHS)
        else:
            epochs = NUM_EPOCHS
        print(f"Number of epochs for {_hospital_name} are {epochs}")
        fed_dict[_hospital_name][round_idx] = {}

        for epoch in range(epochs):
            # Run the training steps on the local dataset
            for imgs, labels in train_dataset:
                train_loss = self.hospitals[_hospital_name].model.train_step(imgs, labels)
            mean_train_loss = np.mean(train_loss)
            print(f"Epoch {epoch + 1}:\tTrain Loss={mean_train_loss:.4f}")
            fed_dict[_hospital_name][round_idx][epoch] = [str(mean_train_loss)]

        print()
        print(f"Evaluation for device {_hospital_name}")
        print('Computing predictions....')
        labels_y_test = list(self.test_dataset.unbatch().map(lambda x, y: tf.argmax(y)))
        results = self.hospitals[_hospital_name].model.predict(self.test_dataset.map(lambda x, y: x))
        y_predicted = list(map(np.argmax, results))

        from sklearn.metrics import f1_score, accuracy_score  # Ensure these are imported
        f1_value = f1_score(labels_y_test, y_predicted, average='macro')
        accuracy_value = accuracy_score(labels_y_test, y_predicted)
        print(f'Accuracy: {accuracy_value:.3f}\tMacro-F1: {f1_value:.3f}')
        print()
        print_line("*")

        # Get the model weights
        self.hospitals[_hospital_name].weights = self.hospitals[_hospital_name].model.get_weights()
        weights = self.hospitals[_hospital_name].weights
        weights_bytes = weights_encoding(weights)
        
        # Upload the weights to IPFS
        start_time = time.time()
        add_info = self.IPFS_client.add(weights_bytes, pin=PIN_BOOL)
        print("IPFS 'add' time: ", str(time.time() - start_time))
        print("IPFS 'add' info: ", add_info.keys())

        # Send the IPFS hash to the Blockchain
        hash_encoded = add_info["Hash"].encode("utf-8")
        send_weights_tx = self.FL_contract.send_weights(
            hash_encoded,
            {"from": self.hospitals[_hospital_name].address},
        )
        self.gas_fee_collab[_hospital_name]['send_fee'].append(send_weights_tx.gas_used)
        send_weights_tx.wait(1)

        return fed_dict

    def retrieving_aggreagted_weights(self, _hospital_name):
        # Retrieve the IPFS hash of the aggregated weights from the Blockchain
        retrieve_aggregated_weights_tx = self.FL_contract.retrieve_aggregated_weights(
            {"from": self.hospitals[_hospital_name].address}
        )
        print(retrieve_aggregated_weights_tx)
        # Optionally record gas usage:
        # self.gas_fee_collab[_hospital_name]['retrieve_fee'].append(retrieve_aggregated_weights_tx.gas_used)
        # retrieve_aggregated_weights_tx.wait(1)
        weight_hash = decode_utf8(retrieve_aggregated_weights_tx, view=True)

        # Download the aggregated weights from IPFS
        start_time = time.time()
        aggregated_weights_encoded = self.IPFS_client.cat(weight_hash)
        print("IPFS 'cat' time: ", str(time.time() - start_time))
        aggregated_weights = weights_decoding(aggregated_weights_encoded)

        # Set the model's weights to the new aggregated weights
        self.hospitals[_hospital_name].aggregated_weights = aggregated_weights
        if isinstance(self.hospitals[_hospital_name].model, FedProx):
            print("Restore weights setting aggregator_weights: FEDPROX")
            FedProx.SERVER_WEIGHTS = aggregated_weights
        else:  # FedAvg
            print("Restore weights setting the weights of aggregator: FEDAVG")
            self.hospitals[_hospital_name].model.set_weights(aggregated_weights)

    async def main(self):
        # Subscribe to the "CloseState" event and set up its alert callback
        self.contract_events.subscribe("CloseState", self.closeState_alert, delay=0.5)

        # Wait for the START event
        coroutine_start = self.contract_events.listen("StartState")
        print("waiting start event...\n")
        await coroutine_start
        print("I waited START")
        print_line("*")
        print('\n' * 2)

        # Begin the start event process
        self.start_event()

        # Wait for the LEARNING event
        coroutine_learning = self.contract_events.listen("LearningState")
        print("waiting learning...\n")
        await coroutine_learning
        print("I waited LEARNING")
        print_line("*")
        print('\n' * 2)

        # Initialization of weights: choose a hospital (the first one) to check the model type
        hosp_name = list(self.hospitals.keys())[0]
        if isinstance(self.hospitals[hosp_name].model, FedProx):
            print("FedProx model weights initialization...")
            global_model = FedAvg(num_classes=4)
            global_model.build((None, WIDTH, HEIGHT, DEPTH))
            global_model.compile(optimizer="adam", metrics="accuracy")
            weights = global_model.trainable_weights
            assert len(weights) != 0
            FedProx.SERVER_WEIGHTS = weights
            file = f'FedProx{MU}_' + self.name
        else:
            file = 'FedAvg_' + self.name

        dataset = self.hospitals[hosp_name].dataset_name
        file_name = dataset + '/' + file

        # Start the Federated Learning loop (which ends when the blockchain CLOSE event is triggered)
        round_idx = 0
        fed_dict = {}
        while True:
            print("Start round loop ...")
            fed_dict = self.round_loop(round_idx, fed_dict, file_name)

            # Wait for the Manager to send the aggregated weights
            coroutine_AW = self.contract_events.listen("AggregatedWeightsReady")
            print("awaiting aggregated weights...\n")
            await coroutine_AW
            print("Aggregated weights arrived!")
            print_line("*")
            print('\n' * 2)

            with open(f"gas_consumption/{file}_collaborator.json", 'w') as json_file:
                json.dump(self.gas_fee_collab, json_file)

            # Continue after reception
            self.aggregatedWeightsReady_event(round_idx)
            round_idx += 1


if __name__ == '__main__':
    collaborator = Collaborator()
    asyncio.run(collaborator.main())
