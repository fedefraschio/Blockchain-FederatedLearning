import os
import sys

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import get_X_test, get_y_test, print_line, set_reproducibility, get_hospitals, load_dataset
from utils_manager import *

from brownie import FederatedLearning, network, accounts
from deploy_FL import get_account
import ipfshttpclient

from constants import *
from sklearn.metrics import classification_report
import numpy as np
import asyncio
import time
import pickle
import tensorflow as tf
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

set_reproducibility()

# connect to IPFS and Blockchain
IPFS_client = ipfshttpclient.connect()
FL_contract = FederatedLearning[-1]
manager = get_account()

# manage contract events
contract_events = FL_contract.events

# storing the overall performance results through the Federated Learning rounds
FL_evaluation = []
FL_classification_report = []

"""
# for SIMULATION/EVALUATION purposes:   - Observe FL model evaluation performance only on SIMULATION env
#                                       - On PRODUCTION env the Manager cannot afford sensitive (test) data from the collaborators
"""

"""
IMPORTANT:
Parameter setting
"""
model_test = FedAvg(NUM_CLASSES)  # creation of the global model always FedAvg, only useful to store weights
model_test.compile(**compile_info)
model_test.build((None, WIDTH, HEIGHT, DEPTH))
hospitals = get_hospitals()
hospital_dataset = load_dataset(hospitals)

test_dataset = hospital_dataset['test']

with open('devices_out_of_battery.pkl', 'rb') as file:
    loaded_list = pickle.load(file)

with open('gas_fee_setup.json', 'r') as json_file:
    gas_fee_manager = json.load(json_file)

model_used = "FedAvg"  # model used by collaborators
if "FedProx" in sys.argv:
    model_used = "FedProx"
    file_name = f"FedProx{MU}_{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{loaded_list}"
else:
    file_name = f"FedAvg_{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{loaded_list}"

hospital_name = list(hospitals.keys())[0]
dataset = hospitals[hospital_name].dataset_name

labels = LABELS_ALZ if dataset == ALZHEIMER else LABELS_TUMOR


def retrive_information():
    # retrieving the parameters IPFS hashes loaded by the collaborators
    hospitals_addresses = FL_contract.get_collaborators({"from": manager})
    retrieved_weights_hash = {}
    for hospital_address in hospitals_addresses:
        retrieved_weights_hash[hospital_address] = FL_contract.retrieve_weights(
            hospital_address, {"from": manager}
        )
    hospitals_weights = {}

    # retrieving the collaborators weights from IPFS
    for hospital_address in retrieved_weights_hash:
        weights_hash = retrieved_weights_hash[hospital_address].decode("utf-8")
        if weights_hash == "":
            print(f"I did not receive anything by: {hospital_address}")
        else:
            start_time = time.time()
            weights_encoded = IPFS_client.cat(weights_hash)
            print("IPFS 'cat' time:", str(time.time() - start_time))
            weights = weights_decoding(weights_encoded)
            hospitals_weights[hospital_address] = weights  # inserting weights only for whom send the weights
        # print_listed_weights(hospitals_weights[hospital_address])

    hospitals_number = len(hospitals_weights)
    weights_dim = len(hospitals_weights[list(hospitals_weights.keys())[0]])
    return weights_dim, hospitals_weights, hospitals_number, hospitals_addresses


def test_information(aggregated_weights):
    """
        function to obtain information about the global model
    """
    model_test.set_weights(aggregated_weights)
    results = model_test.predict(test_dataset.map(lambda x, y: x))
    y_predicted = list((map(np.argmax, results)))
    labels_y_test = list(test_dataset.unbatch().map(lambda x, y: (tf.argmax(y))))

    report = classification_report(
        labels_y_test,
        y_predicted,
        target_names=labels,
        zero_division=False,
        output_dict=True
    )

    FL_classification_report.append(report)
    f1_value = report['macro avg']['f1-score']
    print()
    print("Evaluation of the global model")
    print(f"Accuracy: {report['accuracy']:.3f}\tMacro-F1: {f1_value:.3f} ")
    print()
    return f1_value


def federated_learning():
    weights_dim, hospitals_weights, hospitals_number, hospitals_addresses = retrive_information()
    # computing the AVERAGE of the collaborators weights
    averaged_weights = []

    for i in range(weights_dim):
        layer_weights = []
        for hospital_address in hospitals_weights:
            layer_weights.append(hospitals_weights[hospital_address][i])
        averaged_weights.append(sum(layer_weights) / hospitals_number)

    for i in range(len(averaged_weights)):
        averaged_weights[i] = np.array(
            averaged_weights[i]
        )  # convert the list to a NumPy array

    """
    for hospital_address in hospitals_addresses:
        print_weights(hospitals_weights[hospital_address])
    print_weights(averaged_weights)
    
    # computing the similarity factors
    
    factors = dict.fromkeys(hospitals_addresses, 0)
    for hospital_address in hospitals_addresses:
        if SIMILARITY == 'single':
            factors[hospital_address] = similarity_factor_single(
                hospital_address, hospitals_weights, averaged_weights, hospitals_addresses
            )
        else:
            factors[hospital_address] = similarity_factor_multiple(
                hospital_address, hospitals_weights, averaged_weights, hospitals_addresses
            )
    print("SIMILARITY FACTORS: ")
    for hospital_address in factors:
        print(
            f"Hospital address: {hospital_address}\tSimilarity factor: {factors[hospital_address]}"
        )
    
    # computing the AGGREGATION of the collaborators weights
    aggregated_weights = []

    for i in range(weights_dim):
        layer_weights = []
        for hospital_address in hospitals_addresses:
            if SIMILARITY == 'single':
                layer_weights.append(
                    factors[hospital_address] * hospitals_weights[hospital_address][i]
                )
            elif SIMILARITY == 'multiple':
                layer_weights.append(
                    factors[hospital_address][i] * hospitals_weights[hospital_address][i]
                )
        aggregated_weights.append(sum(layer_weights))
    
    for i in range(len(aggregated_weights)):
        aggregated_weights[i] = np.array(
            aggregated_weights[i]
        )  # Convert the list to a NumPy array
    """
    # show the aggregated weights structure
    # print_weights(aggregated_weights)

    # for TEST purposes:    compare AGGREGATED and AVERAGED parameters performance
    aggregated_weights = averaged_weights

    # sending the aggregated parameters on the IPFS and Blockchain
    aggregated_weights_bytes = weights_encoding(aggregated_weights)
    res = IPFS_client.add(aggregated_weights_bytes, pin=PIN_BOOL)

    hash_encoded = res["Hash"].encode("utf-8")
    send_aggregated_weights_tx = FL_contract.send_aggregated_weights(
        hash_encoded, {"from": manager}
    )
    gas_fee_manager['send_aggregated_weights_fee'].append(send_aggregated_weights_tx.gas_used)
    send_aggregated_weights_tx.wait(1)

    f1_value = test_information(aggregated_weights)
    return f1_value


async def starting():
    print("I am the Manager")
    """uploading model and compile information on the Blockchain"""
    encoded_model = get_encoded_model(NUM_CLASSES, "FedProx")
    print("after get_encoded_model")
    transaction_options = {
        "from": manager,
        "gas_limit": 2000000
    }

    send_model_tx = FL_contract.send_model(encoded_model, transaction_options)
    send_model_tx.wait(1)
    print(send_model_tx.events)

    encoded_compile_info = get_encoded_compile_info()

    send_compile_info_tx = FL_contract.send_compile_info(
        encoded_compile_info, {"from": manager}
    )
    send_compile_info_tx.wait(1)
    print(send_compile_info_tx.events)

    # print model details
    model_test.summary()

    # change the contract state to START
    start_tx = FL_contract.start({"from": manager})
    start_tx.wait(1)
    print(start_tx.events)

    # await for the collaborators to retrieve the model
    coroutine_RM = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    print("COROUTINE: waiting 'retrieve_model'\n", coroutine_RM)
    coroutine_result_PM = await coroutine_RM
    assert_coroutine_result(coroutine_result_PM, "retrieve_model")
    print("I waited retrieve_model")
    print_line("_")

    # await for the collaborators to retrieve the compile information
    coroutine_RCI = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    print("COROUTINE: waiting 'retrieve_compile_info'\n", coroutine_RCI)
    coroutine_result_RCI = await coroutine_RCI
    assert_coroutine_result(coroutine_result_RCI, "retrieve_compile_info")
    print("I waited retrieve_compile_info")
    print_line("_")

    # hospitals synchronization
    time.sleep(10)

    # change the contract state to LEARNING
    learning_tx = FL_contract.learning({"from": manager})
    learning_tx.wait(1)
    # print(learning_tx.events)


async def main():
    best_f1 = 0.
    best_model = None
    gas_fee_manager['send_aggregated_weights_fee'] = []

    '''uploading model and compile information on the Blockchain'''

    print('uploading model and compile information on the Blockchain')
    encoded_model = get_encoded_model(NUM_CLASSES, model_used)

    transaction_options = {
        "from": manager,
        "gas_limit": 2000000
    }

    send_model_tx = FL_contract.send_model(encoded_model, transaction_options)
    gas_fee_manager['send_model_fee'] = send_model_tx.gas_used
    send_model_tx.wait(1)

    encoded_compile_info = get_encoded_compile_info()
    send_compile_info_tx = FL_contract.send_compile_info(
        encoded_compile_info, {"from": manager}
    )
    gas_fee_manager['send_model_fee'] += send_compile_info_tx.gas_used
    send_compile_info_tx.wait(1)

    print_line("*")
    print('\n' * 2)

    '''print model details'''
    print('model details')
    model_test.summary()
    print_line("*")
    print('\n' * 2)

    '''change the contract state to START'''
    print('change the contract state to START')
    print()
    start_tx = FL_contract.start({"from": manager})
    gas_fee_manager['change_state_fee'] = start_tx.gas_used
    start_tx.wait(1)
    print_line("*")
    print('\n' * 2)

    '''await for the collaborators to retrieve the model'''
    print('await for the collaborators to retrieve the model...')
    print()
    print()
    coroutine_RM = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    coroutine_result_PM = await coroutine_RM
    assert_coroutine_result(coroutine_result_PM, "retrieve_model")
    print("I waited retrieve_model")
    print_line("*")
    print('\n' * 2)

    '''await for the collaborators to retrieve the compile information'''
    print('await for the collaborators to retrieve the compile information...')
    print()
    coroutine_RCI = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    coroutine_result_RCI = await coroutine_RCI
    assert_coroutine_result(coroutine_result_RCI, "retrieve_compile_info")
    print("I waited retrieve_compile_info")
    print_line("*")
    print('\n' * 2)

    '''hospitals synchronization'''
    time.sleep(10)

    '''change the contract state to LEARNING'''
    print('change the contract state to LEARNING')
    print()
    learning_tx = FL_contract.learning({"from": manager})
    gas_fee_manager['change_state_fee'] += learning_tx.gas_used
    learning_tx.wait(1)
    print_line("*")
    print('\n' * 2)


    for round in range(NUM_ROUNDS):
        print(f"\t\tFL ROUND {round + 1}...")

        # await for the collaborators to send the weights
        coroutine_SW = contract_events.listen(
            "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_DEVICES
        )
        coroutine_result_SW = await coroutine_SW
        # assert_coroutine_result(coroutine_result_SW, "send_weights")
        print("Weights arrived")
        print_line("*")

        # reset the weights related events
        reset_weights_tx = FL_contract.reset_weights({"from": manager})
        gas_fee_manager['change_state_fee'] += reset_weights_tx.gas_used
        reset_weights_tx.wait(1)

        # continue after reception
        f1_value = federated_learning()
        if best_f1 < f1_value:
            best_f1 = f1_value
            best_model = model_test
        print_line("*")

    # close the BLockchain at the end of the Federated Learning
    close_tx = FL_contract.close({"from": manager})
    gas_fee_manager['change_state_fee'] += close_tx.gas_used
    close_tx.wait(1)

    network.disconnect()

    with open(f"gas_consumption/{file_name}_manager.json", 'w') as json_file:
        json.dump(gas_fee_manager, json_file)

    """
    print("RESULTS - Overall Performance Evaluation through Federated Learning...")
    acc = {}
    for round in range(NUM_ROUNDS):
        print(FL_classification_report[round])
    
    
    print("Saving best model ...")
    path = f"./models/{dataset}/" + file_name + '/'
    if not os.path.exists(path):
        # If it doesn't exist, create it
        os.makedirs(path)
    best_model.save_weights(path)
    """
    sys.exit(0)


asyncio.run(main())
