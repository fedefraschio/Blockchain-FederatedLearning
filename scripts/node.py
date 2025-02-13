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

# Choose the initial role via a command-line argument or configuration.
# For example: python node.py aggregator
if len(sys.argv) > 1 and sys.argv[3].lower() == "aggregator":
    initial_role = "aggregator"
else:
    initial_role = "collaborator"

hospital_name = sys.argv[4]

# -----------------------
# Aggregator-specific logic
# -----------------------

async def aggregator_mode():
    print(">>> Running as Aggregator")
    # Place here (or call) the asynchronous aggregator logic from your aggregator script.
    # For example:
    # - Send the model and compile info to the blockchain.
    # - Wait for collaborators to retrieve them.
    # - Wait for the collaborators to send back their weights.
    # - Retrieve weights, aggregate them, and send back the aggregated weights.
    # - After completing a round, signal role transfer on the blockchain.
    mgr = Manager()
    await mgr.main()
    print(">>> Aggregator round complete; passing role to collaborator")

# -----------------------
# Collaborator-specific logic
# -----------------------
async def collaborator_mode():
    print(">>> Running as Collaborator")
    # Place here (or call) the asynchronous collaborator logic from your collaborator script.
    # For example:
    # - Listen for the START event, then retrieve model/compile info.
    # - Train the model on local data.
    # - Upload your weights and wait for aggregated weights.
    collab = Collaborator(hospital_name=hospital_name, out_of_battery=False, network=None)
    await collab.main()
    print(">>> Collaborator round complete; waiting to see if I become aggregator")

# -----------------------
# Role-transfer watcher
# -----------------------
async def watch_for_role_transfer():
    # This function listens for a blockchain (or other) event
    # that tells this node it should become the aggregator.
    ######await asyncio.sleep(15)  # simulate waiting for the event # TO BE INSERTED AGAIN
    coroutine_transfer = contract_events.listen("NewAggregatorElected")
    coroutine_result_transfer = await coroutine_transfer
    aggregator = coroutine_result_transfer["args"]["aggregator"]
    print("New aggregator elected:", aggregator)
    ###
    '''
    TO DO: if aggregator == self: return 'aggregator' else return 'collaborator
    '''
    ###
    return "aggregator"

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
            # After finishing a round, you might decide to pass the role.
            # (For example, your aggregator routine could have sent a blockchain event that
            # tells another node to become aggregator. Your node, after finishing,
            # could choose to become a collaborator.)
            role = "collaborator"
        else:
            # In collaborator mode, run the collaborator task concurrently with a watcher.
            # It does not block execution.
            # It returns a task object that represents the running coroutine.
            collab_task = asyncio.create_task(collaborator_mode())
            role_watcher = asyncio.create_task(watch_for_role_transfer())

            # Wait until one of the tasks finishes.
            # 'done' will contain the completed task(s), and 'pending' will contain the ones still running.
            done, pending = await asyncio.wait(
                [collab_task, role_watcher],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if role_watcher in done:
                # The node is being signaled to switch to aggregator mode.
                role = role_watcher.result()  # expected to be "aggregator"
                # Optionally cancel the collaborator work if it’s still running:
                for task in pending:
                    task.cancel()
            else:
                # Otherwise, keep being a collaborator (or perform other logic).
                role = "collaborator"
        

        # Optionally, you can add a break condition (e.g., when FL is complete).
        # For this example, we let it run indefinitely.
        print(f"Switching role; current role is now '{role}'\n")

# -----------------------
# Entry point
# -----------------------

# Run the node main loop with asyncio.
asyncio.run(node_main(initial_role))

