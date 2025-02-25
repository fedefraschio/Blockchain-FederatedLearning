import asyncio
import sys

import os
import sys

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import get_X_test, get_y_test, print_line, set_reproducibility, get_hospitals, load_dataset
from utils_manager import *
from new_manager import Manager
from new_collaborator import Collaborator

from brownie import FederatedLearning, network, accounts
from deploy_FL import get_account
import ipfshttpclient

from sklearn.metrics import classification_report
import numpy as np
import asyncio
import time
import pickle
import tensorflow as tf
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

set_reproducibility()

# Connect to IPFS and the Blockchain contract
FL_contract = FederatedLearning[-1]
contract_events = FL_contract.events

# Retrieve hospitals and load dataset
hospitals = get_hospitals()

# Choose the initial role via a command-line argument or configuration.
# For example: python node.py aggregator

if len(sys.argv) != 6 and len(sys.argv) != 7:
    raise ValueError("Invalid number of arguments")

if len(sys.argv) > 1 and sys.argv[4].lower() == "aggregator":
    initial_role = "aggregator"
else:
    initial_role = "collaborator"

hospital_name = sys.argv[3]

hospital_name = sys.argv[3]

async def aggregator_mode():
    print(">>> Running as Aggregator")

    mgr = Manager()
    
    await mgr.main()
    print(">>> Aggregator round complete; passing role to collaborator")

# -----------------------
# Collaborator-specific logic
# -----------------------
async def collaborator_mode():
    print(">>> Running as Collaborator")
    
    collab = Collaborator(hospital_name=hospital_name, out_of_battery=False, network=None)
    
    # Collaborator runs util the exception is raised, in that case, we handle the role changing
    try:
        await collab.main()  # This should support graceful stopping
    except asyncio.CancelledError:
        print(">>> Collaborator execution was cancelled")
        print("My address is: " + str(hospitals[hospital_name].address))
        print("The next aggregator will be: " + str(FL_contract.get_aggregator()))
        print("All the collaborators are: " + str(FL_contract.get_collaborators()))
        return
    print(">>> Collaborator round complete; waiting to see if I become aggregator")

# -----------------------
# Role-transfer watcher
# -----------------------
'''
async def watch_for_role_transfer():
    # This function listens for a blockchain (or other) event
    # that tells this node it should become the aggregator.
    # For example, you might subscribe to an event "AggregatorRoleTransfer"
    # and check if the new aggregator address matches this node's address.
    # Here, we simply simulate an event after a delay.
    await asyncio.sleep(15)  # simulate waiting for the event
    # In a real implementation, return a value or set a flag.
    print(">>> Received event: become aggregator")
    return "aggregator"
'''
# -----------------------
# Main node logic that manages role switching
# -----------------------

async def node_main(initial_role: str):
    role = initial_role
    # Run forever (or for the number of rounds you need)
    while True:
        if role == "aggregator":
            # Run aggregator tasks.
            await aggregator_mode()
            role = "collaborator"
        else:
            await collaborator_mode()

        # Optionally, you can add a break condition (e.g., when FL is complete).
        # For this example, we let it run indefinitely.
        print(f"Switching role; current role is now '{role}'\n")

# -----------------------
# Entry point
# -----------------------

# Run the node main loop with asyncio.
asyncio.run(node_main(initial_role))

