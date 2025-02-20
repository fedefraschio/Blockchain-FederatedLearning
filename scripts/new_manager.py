import os
import sys
import asyncio
import json
import time
import pickle
import numpy as np
import tensorflow as tf
import ipfshttpclient

# Brownie and related imports
from brownie import FederatedLearning, network, accounts
from deploy_FL import get_account

# Your utility modules and constants
from utils_simulation import get_X_test, get_y_test, print_line, set_reproducibility, get_hospitals, load_dataset
from utils_manager import *  # (Ensure you import only what you need)
from constants import *
from sklearn.metrics import classification_report

# Suppress TF logging and set reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_reproducibility()

# Make sure the directory containing this script is in sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)


class Manager:
    def __init__(self):
        # Initialization flag
        self.first_run = True

        # Connect to IPFS and Blockchain
        self.IPFS_client = ipfshttpclient.connect()
        self.FL_contract = FederatedLearning[-1]
        self.manager = get_account()
        self.contract_events = self.FL_contract.events

        # Containers for performance evaluation
        self.FL_evaluation = []
        self.FL_classification_report = []

        # Create the global model (FedAvg is assumed; adjust if needed)
        self.model_test = FedAvg(NUM_CLASSES)
        self.model_test.compile(**compile_info)
        self.model_test.build((None, WIDTH, HEIGHT, DEPTH))

        # Load hospitals and dataset information
        self.hospitals = get_hospitals()
        self.hospital_dataset = load_dataset(self.hospitals)
        self.test_dataset = self.hospital_dataset['test']

        # Ensure that the weights are retrieved just once from previous iteration
        self.get_previous_weigths = False

        # Load extra configuration
        with open('devices_out_of_battery.pkl', 'rb') as file:
            self.loaded_list = pickle.load(file)

        with open('gas_fee_setup.json', 'r') as json_file:
            self.gas_fee_manager = json.load(json_file)

        # Determine the model type and file naming scheme
        self.model_used = "FedAvg"
        if "FedProx" in sys.argv:
            self.model_used = "FedProx"
            self.file_name = f"FedProx{MU}_{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{self.loaded_list}"
        else:
            self.file_name = f"FedAvg_{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{self.loaded_list}"

        # Set additional parameters
        self.hospital_name = list(self.hospitals.keys())[0]
        self.dataset = self.hospitals[self.hospital_name].dataset_name
        self.labels = LABELS_ALZ if self.dataset == ALZHEIMER else LABELS_TUMOR

    def retrieve_information(self):
        """Retrieves weights from collaborators via the blockchain and IPFS."""
        hospitals_addresses = self.FL_contract.get_collaborators({"from": self.manager})
        retrieved_weights_hash = {}
        for hospital_address in hospitals_addresses:
            retrieved_weights_hash[hospital_address] = self.FL_contract.retrieve_weights(
                hospital_address, {"from": self.manager}
            )
        hospitals_weights = {}
        # Retrieve weights from IPFS for each collaborator
        for hospital_address in retrieved_weights_hash:
            weights_hash = retrieved_weights_hash[hospital_address].decode("utf-8")
            if weights_hash == "":
                print(f"I did not receive anything from: {hospital_address}")
            else:
                start_time = time.time()
                weights_encoded = self.IPFS_client.cat(weights_hash)
                print("IPFS 'cat' time:", str(time.time() - start_time))
                weights = weights_decoding(weights_encoded)
                hospitals_weights[hospital_address] = weights
        hospitals_number = len(hospitals_weights)
        # Assume all collaborators return weights with the same dimensionality
        weights_dim = len(hospitals_weights[list(hospitals_weights.keys())[0]])
        return weights_dim, hospitals_weights, hospitals_number, hospitals_addresses

    def test_information(self, aggregated_weights):
        """Evaluates the global model using aggregated weights and prints the performance report."""
        self.model_test.set_weights(aggregated_weights)
        results = self.model_test.predict(self.test_dataset.map(lambda x, y: x))
        y_predicted = list(map(np.argmax, results))
        labels_y_test = list(self.test_dataset.unbatch().map(lambda x, y: tf.argmax(y)))
        report = classification_report(
            labels_y_test,
            y_predicted,
            target_names=self.labels,
            zero_division=False,
            output_dict=True
        )
        self.FL_classification_report.append(report)
        f1_value = report['macro avg']['f1-score']
        print()
        print("Evaluation of the global model")
        print(f"Accuracy: {report['accuracy']:.3f}\tMacro-F1: {f1_value:.3f}")
        print()
        return f1_value

    def federated_learning(self):
        """Performs federated aggregation of collaborator weights and sends the aggregated weights."""
        weights_dim, hospitals_weights, hospitals_number, hospitals_addresses = self.retrieve_information()
        # Compute the average of the collaborators' weights
        averaged_weights = []
        for i in range(weights_dim):
            layer_weights = []
            for hospital_address in hospitals_weights:
                layer_weights.append(hospitals_weights[hospital_address][i])
            averaged_weights.append(sum(layer_weights) / hospitals_number)
        # Convert each averaged layer into a NumPy array
        for i in range(len(averaged_weights)):
            averaged_weights[i] = np.array(averaged_weights[i])
        # For testing purposes, we use the averaged weights as the aggregated weights
        aggregated_weights = averaged_weights

        ###### TEST ######
        # 
        aggregated_weights_before = self.FL_contract.retrieve_aggregated_weights({"from": self.manager})
        if int(str(aggregated_weights_before), 16) != 0  and self.get_previous_weigths == False:
            print("Getting aggregated weights previous aggregator")
            weight_hash = decode_utf8(aggregated_weights_before, view=True)

            # Download the aggregated weights from IPFS
            start_time = time.time()
            aggregated_weights_encoded = self.IPFS_client.cat(weight_hash)
            print("IPFS 'cat' time: ", str(time.time() - start_time))
            aggregated_weights = weights_decoding(aggregated_weights_encoded)
            self.get_previous_weigths = True
        ###### #### ######

        # Upload the aggregated weights to IPFS
        aggregated_weights_bytes = weights_encoding(aggregated_weights)

        res = self.IPFS_client.add(aggregated_weights_bytes, pin=PIN_BOOL)
        hash_encoded = res["Hash"].encode("utf-8")
        # Send the aggregated weights to the blockchain
        send_aggregated_weights_tx = self.FL_contract.send_aggregated_weights(
            hash_encoded, {"from": self.manager}
        )
        self.gas_fee_manager['send_aggregated_weights_fee'].append(send_aggregated_weights_tx.gas_used)
        send_aggregated_weights_tx.wait(1)

        f1_value = self.test_information(aggregated_weights)
        return f1_value

    async def starting(self):
        """Uploads model/compile info, starts the contract and waits for collaborators to retrieve the data."""
        print("I am the Manager")
        # Upload model
        encoded_model = get_encoded_model(NUM_CLASSES, "FedProx")
        print("after get_encoded_model")
        transaction_options = {"from": self.manager, "gas_limit": 2000000}
        send_model_tx = self.FL_contract.send_model(encoded_model, transaction_options)
        send_model_tx.wait(1)
        print(send_model_tx.events)

        # Upload compile information
        encoded_compile_info = get_encoded_compile_info()
        send_compile_info_tx = self.FL_contract.send_compile_info(
            encoded_compile_info, {"from": self.manager}
        )
        send_compile_info_tx.wait(1)
        print(send_compile_info_tx.events)

        # Print model details
        self.model_test.summary()

        # Change the contract state to START
        start_tx = self.FL_contract.start({"from": self.manager})
        start_tx.wait(1)
        print(start_tx.events)

        # Wait for collaborators to retrieve the model and compile information
        coroutine_RM = self.contract_events.listen("EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS)
        print("COROUTINE: waiting 'retrieve_model'\n", coroutine_RM)
        coroutine_result_PM = await coroutine_RM
        assert_coroutine_result(coroutine_result_PM, "retrieve_model")
        print("I waited retrieve_model")
        print_line("_")

        coroutine_RCI = self.contract_events.listen("EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS)
        print("COROUTINE: waiting 'retrieve_compile_info'\n", coroutine_RCI)
        coroutine_result_RCI = await coroutine_RCI
        assert_coroutine_result(coroutine_result_RCI, "retrieve_compile_info")
        print("I waited retrieve_compile_info")
        print_line("_")

        # Synchronization pause
        time.sleep(10)

        # Change the contract state to LEARNING
        learning_tx = self.FL_contract.learning({"from": self.manager})
        learning_tx.wait(1)

    async def main(self):
        """Main asynchronous flow of the Manager."""
        best_f1 = 0.0
        best_model = None
        self.gas_fee_manager['send_aggregated_weights_fee'] = []

        # Ensure that the blockchain is open
        open_tx = self.FL_contract.open({"from": self.manager})
        self.gas_fee_manager['open_blockchain_fee'] += open_tx.gas_used
        open_tx.wait(1)

        # Upload model and compile information on the Blockchain
        print('Uploading model and compile information on the Blockchain')
        encoded_model = get_encoded_model(NUM_CLASSES, self.model_used)
        transaction_options = {"from": self.manager, "gas_limit": 2000000}
        send_model_tx = self.FL_contract.send_model(encoded_model, transaction_options)
        self.gas_fee_manager['send_model_fee'] = send_model_tx.gas_used
        send_model_tx.wait(1)

        encoded_compile_info = get_encoded_compile_info()
        send_compile_info_tx = self.FL_contract.send_compile_info(
            encoded_compile_info, {"from": self.manager}
        )
        self.gas_fee_manager['send_model_fee'] += send_compile_info_tx.gas_used
        send_compile_info_tx.wait(1)

        print_line("*")
        print('\n' * 2)

        # Print model details
        print('Model details')
        self.model_test.summary()
        print_line("*")
        print('\n' * 2)

        # Change the contract state to START
        print('Changing the contract state to START')
        start_tx = self.FL_contract.start({"from": self.manager})
        self.gas_fee_manager['change_state_fee'] = start_tx.gas_used
        start_tx.wait(1)
        print_line("*")
        print('\n' * 2)

        # Wait for collaborators to retrieve the model
        print('Awaiting for the collaborators to retrieve the model...')
        coroutine_RM = self.contract_events.listen("EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS)
        coroutine_result_PM = await coroutine_RM
        assert_coroutine_result(coroutine_result_PM, "retrieve_model")
        print("I waited retrieve_model")
        print_line("*")
        print('\n' * 2)

        # Wait for collaborators to retrieve the compile information
        print('Awaiting for the collaborators to retrieve the compile information...')
        coroutine_RCI = self.contract_events.listen("EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS)
        coroutine_result_RCI = await coroutine_RCI
        assert_coroutine_result(coroutine_result_RCI, "retrieve_compile_info")
        print("I waited retrieve_compile_info")
        print_line("*")
        print('\n' * 2)

        # Synchronization pause
        time.sleep(10)

        # Change the contract state to LEARNING
        print('Changing the contract state to LEARNING')
        learning_tx = self.FL_contract.learning({"from": self.manager})
        self.gas_fee_manager['change_state_fee'] += learning_tx.gas_used
        learning_tx.wait(1)
        print_line("*")
        print('\n' * 2)

        # Start the federated learning rounds
        for round in range(NUM_ROUNDS):
            print(f"\t\tFL ROUND {round + 1}...")
            coroutine_SW = self.contract_events.listen("EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_DEVICES)
            coroutine_result_SW = await coroutine_SW
            print("Weights arrived")
            print_line("*")

            reset_weights_tx = self.FL_contract.reset_weights({"from": self.manager})
            self.gas_fee_manager['change_state_fee'] += reset_weights_tx.gas_used
            reset_weights_tx.wait(1)

            f1_value = self.federated_learning()
            if best_f1 < f1_value:
                best_f1 = f1_value
                best_model = self.model_test
            print_line("*")

        # Close the blockchain process at the end of Federated Learning
        close_tx = self.FL_contract.close({"from": self.manager})
        self.gas_fee_manager['change_state_fee'] += close_tx.gas_used
        close_tx.wait(1)

        # Pass role to the other collaborator
        self.FL_contract.electNewAggregator({"from": self.manager})

        #####network.disconnect()

        # Save gas consumption data
        with open(f"gas_consumption/{self.file_name}_manager.json", 'w') as json_file:
            json.dump(self.gas_fee_manager, json_file)

        # Optionally, you might want to save the best model here
        # sys.exit(0)  # Uncomment if you want the script to exit

