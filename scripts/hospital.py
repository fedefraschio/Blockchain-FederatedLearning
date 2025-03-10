import os
import sys
from numpy import require

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

import ipfshttpclient
import asyncio
import json
import pickle
from utils_simulation import get_hospitals, print_line, set_reproducibility, round_out_of_battery, \
    device_out_of_battery, load_dataset
from utils_collaborator import *
from brownie import FederatedLearning
from classHospital import Hospital
from tensorflow.keras.models import model_from_json
from constants import NUM_ROUNDS
from sklearn.metrics import classification_report, f1_score, accuracy_score
from fedAvg import FedAvg
from fedProx import FedProx

from constants import *
import random
import time
import tensorflow as tf

set_reproducibility()

# retrieve the hospitals information
hospitals = get_hospitals()
hospital_dataset = load_dataset(hospitals)
test_dataset = hospital_dataset['test']

# connect to IPFS and Blockchain
IPFS_client = ipfshttpclient.connect()
FL_contract = FederatedLearning[-1]

# manage contract events
contract_events = FL_contract.events

# storing the hospitals performance results through the Federated Learning rounds
hospitals_evaluation = {hospital_name: [] for hospital_name in hospitals}

ROUND_BATTERY = 100  # setting an higher time to avoid out of battery
DEVICES_OUT_OF_BATTERY = []
file_name = ""
if "out" in sys.argv:
    ROUND_BATTERY = round_out_of_battery(NUM_ROUNDS)
    DEVICES_OUT_OF_BATTERY = device_out_of_battery(hospitals, n=NUM_DEVICES_OUT_BATTERY)
    print(f"Device/s {DEVICES_OUT_OF_BATTERY} will be out of battery at round {ROUND_BATTERY + 1}")
    name = f"{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{DEVICES_OUT_OF_BATTERY}"
else:
    print("No devices out of battery")
    name = f"{NUM_ROUNDS}_{NUM_EPOCHS}_{NUM_DEVICES}_{[]}"

with open('devices_out_of_battery.pkl', 'wb') as file:
    # Use pickle.dump to save the list to the file
    pickle.dump(DEVICES_OUT_OF_BATTERY, file)

# dictionary for collaborator fee
gas_fee_collab = {hospital_name: {'retrieve_fee': [], 'send_fee': [], 'model_start_fee': 0} for hospital_name in
                  hospitals}


def closeState_alert(event):
    print("The FL Blockchain has been CLOSED\n")
    print("RESULTS - Hospitals Performance Evaluation through Federated Learning...")
    for hospital_name in hospitals_evaluation:
        print(f"{hospital_name}:")
        for round, [loss, acc] in enumerate(hospitals_evaluation[hospital_name], start=1):
            print(f"\tRound {round}:\tLoss: {loss:.3f} - Accuracy: {acc:.3f}")
    # network.disconnect()
    # sys.exit(0)


# triggered after the START event from the Blockchain
def start_event():
    for hospital_name in hospitals:
        # retrieving of the model given by the Manager
        retrieve_model_tx = FL_contract.retrieve_model(
            {"from": hospitals[hospital_name].address}
        )
        gas_fee_collab[hospital_name]['model_start_fee'] = retrieve_model_tx.gas_used
        retrieve_model_tx.wait(1)


        custom_objects = {'FedAvg': FedAvg, 'FedProx': FedProx}
        decoded_model = decode_utf8(retrieve_model_tx)
        model = model_from_json(decoded_model, custom_objects=custom_objects)

        print("Model ", model)
        hospitals[hospital_name].model = model

        # retrieving of the compile information goven by the Manager
        retreive_compile_info_tx = FL_contract.retrieve_compile_info(
            {"from": hospitals[hospital_name].address}
        )
        gas_fee_collab[hospital_name]['model_start_fee'] += retreive_compile_info_tx.gas_used
        retreive_compile_info_tx.wait(1)

        decoded_compile_info = decode_utf8(retreive_compile_info_tx)
        fl_compile_info = json.loads(decoded_compile_info)
        hospitals[hospital_name].compile_info = fl_compile_info

        # compiling the model with the compile information
        hospitals[hospital_name].model.compile(**hospitals[hospital_name].compile_info)


# operations to do at every FL round
def round_loop(round, fed_dict, file_name):
    for hospital_name in hospitals:
        if hospital_name not in fed_dict:
            fed_dict[hospital_name] = {}
        if round >= ROUND_BATTERY and hospital_name in DEVICES_OUT_OF_BATTERY:
            print(f"Device {hospital_name} is out of battery")
            fed_dict[hospital_name][round] = "out_of_battery"
        else:
            print(f"Device {hospital_name} is training ...")
            fed_dict = fitting_model_and_loading_weights(hospital_name, round, fed_dict)

    path = './results/' + file_name + '.json'
    with open(path, 'w') as json_file:
        json.dump(fed_dict, json_file)
    return fed_dict


# triggered after the 'aggregatedWeightsReady' event from the Blockchain
def aggregatedWeightsReady_event(round):
    for hospital_name in hospitals:
        if hospital_name in DEVICES_OUT_OF_BATTERY and (round + 1) >= ROUND_BATTERY:
            continue
        print("Retrieving weights for hospital ", hospital_name)
        retrieving_aggreagted_weights(hospital_name)
        print("-" * 50)
        print()


def fitting_model_and_loading_weights(_hospital_name, round, fed_dict):

    train_dataset = hospital_dataset[_hospital_name]
    epochs = random.randint(1, NUM_EPOCHS) if isinstance(hospitals[_hospital_name].model, FedProx) else NUM_EPOCHS
    print(f"Number of epochs for {_hospital_name} are {epochs}")
    fed_dict[_hospital_name][round] = {}

    for epoch in range(epochs):

        for imgs, labels in train_dataset:
            train_loss = hospitals[_hospital_name].model.train_step(imgs, labels)

        mean_train_loss = np.mean(train_loss)
        print(f"Epoch {epoch + 1}:\tTrain Loss={mean_train_loss:.4f}")
        fed_dict[_hospital_name][round][epoch] = [str(mean_train_loss)]

    print()
    print(f"Evaluation for device {_hospital_name}")
    print('Computing predictions....')
    labels_y_test = list(test_dataset.unbatch().map(lambda x, y: (tf.argmax(y))))
    results = hospitals[_hospital_name].model.predict(test_dataset.map(lambda x, y: x))
    y_predicted = list((map(np.argmax, results)))

    f1_value = f1_score(labels_y_test, y_predicted, average='macro')
    accuracy_value = accuracy_score(labels_y_test, y_predicted)
    print(f'Accuracy: {accuracy_value:.3f}\tMacro-F1: {f1_value:.3f}')
    print()
    print_line("*")
    
    # Hospital evaluation (COMMENT this to speed things up)
    '''
    hospitals_evaluation[_hospital_name].append(
        hospitals[_hospital_name].model.evaluate(test_dataset)
    )
    '''
    
    hospitals[_hospital_name].weights = hospitals[_hospital_name].model.get_weights()

    """ loading weights """
    weights = hospitals[_hospital_name].weights
    weights_bytes = weights_encoding(weights)
    # print("weights_JSON size:" + str(sys.getsizeof(weights_JSON)))

    # uploading the weights on IPFS
    start_time = time.time()
    add_info = IPFS_client.add(weights_bytes, pin=PIN_BOOL)
    print("IPFS 'add' time: ", str(time.time() - start_time))
    print("IPFS 'add' info: ", add_info.keys())

    # sending the IPFS hash of the weights in the Blockchain
    hash_encoded = add_info["Hash"].encode("utf-8")
    send_weights_tx = FL_contract.send_weights(
        hash_encoded,
        {"from": hospitals[_hospital_name].address},
    )
    gas_fee_collab[_hospital_name]['send_fee'].append(send_weights_tx.gas_used)
    send_weights_tx.wait(1)

    return fed_dict


def retrieving_aggreagted_weights(_hospital_name):
    # retrieve the IPFS hash of the aggregated wights from the Blockchain
    retrieve_aggregated_weights_tx = FL_contract.retrieve_aggregated_weights(
        {"from": hospitals[_hospital_name].address}
    )
    # gas_fee_collab[_hospital_name]['retrieve_fee'].append(retrieve_aggregated_weights_tx.gas_used)
    # retrieve_aggregated_weights_tx.wait(1)

    weight_hash = decode_utf8(retrieve_aggregated_weights_tx, view=True)

    # download the aggregated weights from IPFS
    start_time = time.time()
    aggregated_weights_encoded = IPFS_client.cat(weight_hash)
    print("IPFS 'cat' time: ", str(time.time() - start_time))

    aggregated_weights = weights_decoding(aggregated_weights_encoded)

    # setting the model with the new aggregated weights computed by the Manager
    hospitals[_hospital_name].aggregated_weights = aggregated_weights
    if isinstance(hospitals[_hospital_name].model, FedProx):
        print("Restore weights setting aggregator_weights: FEDPROX")
        FedProx.SERVER_WEIGHTS = aggregated_weights
    else:  # FedAvg
        print("Restore weights setting the weights of aggregator: FEDAVG")
        hospitals[_hospital_name].model.set_weights(aggregated_weights)


async def main():
    '''continuously listens for the CLOSE event from the Blockchain and promptly handles it'''
    contract_events.subscribe("CloseState", closeState_alert, delay=0.5)

    '''await for the START event'''
    coroutine_start = contract_events.listen("StartState")
    print("waiting start event...")
    print()
    await coroutine_start
    print("I waited START")
    print_line("*")
    print('\n' * 2)

    '''start event begin'''
    start_event()

    '''await for the LEARNING event'''
    coroutine_learning = contract_events.listen("LearningState")
    print("waiting learning...")
    print()
    await coroutine_learning
    print("I waited LEARNING")
    print_line("*")
    print('\n' * 2)

    '''Initialiazation weights'''
    hospital_name = list(hospitals.keys())[0]  # take the first elements to check the model used
    if isinstance(hospitals[hospital_name].model, FedProx):
        print("FedProx model weights initialization...")
        global_model = FedAvg(num_classes=4)
        global_model.build((None, WIDTH, HEIGHT, DEPTH))
        global_model.compile(optimizer="adam", metrics="accuracy")
        weights = global_model.trainable_weights
        assert len(weights) != 0
        FedProx.SERVER_WEIGHTS = weights
        file = f'FedProx{MU}_' + name
    else:
        file = 'FedAvg_' + name

    dataset = hospitals[hospital_name].dataset_name
    file_name = dataset + '/' + file

    '''start of the Federated Learning loop that will be ended by the CLOSE event'''
    round = 0
    fed_dict = {}
    while True:
        print("Start round loop ...")
        fed_dict = round_loop(round, fed_dict, file_name)

        # await for the Manager to send the aggregated weights
        coroutine_AW = contract_events.listen("AggregatedWeightsReady")
        print("awaiting aggregated weights...")
        print()
        await coroutine_AW
        print("Aggregated weights arrived!")
        print_line("*")
        print('\n' * 2)


        with open(f"gas_consumption/{file}_collaborator.json", 'w') as json_file:
            json.dump(gas_fee_collab, json_file)

        # continue after reception
        aggregatedWeightsReady_event(round)
        round += 1


asyncio.run(main())
