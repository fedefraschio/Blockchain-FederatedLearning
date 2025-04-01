# Federated Learning: Simulation of a Peer-2-Peer Federated Learning Architecture with Dynamic Role assignment

Project for the course Blockchain and Cryptocurrencies, at the Alma Mater Studiorum - Università di Bologna. This project is an extension of the following project: https://github.com/LorenzoCassano/Blockchain-FederatedLearning

## Additional features implemented
<ul>
<li><b>Updated the Smart Contract</b>: 
  <ul>
    <li>Checking the access to the contract using AccessControl from OpenZeppelin,</li>
    <li>Added functions for handling aggregator Failure (reportAggregatorFailure(), electNewAggregator(), reportTimeout()),</li>
    <li>Added a function that resets the state of the contract when contract is closed (resetContractState())</li>   
  </ul>
 </li>
<li><b>Parallelized the architecture</b>: the nodes of the architecture are executed on different terminals, to simulate a realistic FL architecture</li>
<li><b>Dynamic role assignment</b>: each node can be independently assigned the role of collaborator or aggregator, depending on what's needed by the system</li>
<li><b>Downloading previously aggregated weights if present</b>: when setting up the model, the collaborators always check if there are any previously aggregated weights on IPFS.</li>
</ul>

## Description of the simulation
We have n nodes: n-1 collaborators and 1 aggregator. During each FL round, each collaborator instantiates the CNN model and starts training. After training for a certain number of epochs, the weights are sent to the aggregator which aggregates them and sends the aggregated weights back to the collaborators. At the end of the FL round the node which has been the aggregator passes the aggregator role to the next node on the list (Round-Robin).

## Setup
This setup is just for a simulation
### Requirements
* Ganache
* IPFS
* Miniconda
  * eth-brownie
  * cuda
  * tensorflow
  * opencv-python
  * pandas
  * scikit-learn

### base deactivate
`conda deactivate`

### environment creation
`conda create --name blockchain_project python=3.9`

### activation
`conda activate blockchain_project`

### pip update
`python -m install pip --upgrade pip`

### cuda installation
`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

### tensorflow=2.10 installation
`pip install "tensorflow<2.11"`

### opencv-python installation
`pip install opencv-python`

### pandas installation
`pip install pandas==1.5.3`

### eth-brownie installation
`pip install eth-brownie`

### scikit-learn installation
`pip install scikit-learn`

### ganache installation
https://trufflesuite.com/ganache/

### ipfs installation
https://github.com/ipfs/ipfs-desktop/releases

### add network brownie
`brownie networks add Ethereum fl-local host=http://127.0.0.1:7545 chainid=5777 timeout=3600`

### check network
`brownie networks list`

## Running
This is just a simulation. For concurruncy problems on training on the same GPU, the _collaborator.py_ script contains a loop that trains the
different hospital model instances one at time in sequence. In a real time scenario, with more than one peer, it is possible to run 
the different learnings at the same time and it works in the same way.

### Settings of the simulation
All the parameters of the simulation can be set in the file _constants.py_
- Number of devices (nodes)
- FL rounds
- Number of epochs
- ...

### Step 1: Setup
#### Setup first time
It is possible to choose the dataset, inserting the parameter, "brain_tumor" it will be used the brain tumor dataset, if the dataset is not specified it will be used the Alzheimer dataset

For **Brain Tumor**:

`brownie run .\scripts\setup.py main brain_tumor --network fl-local` 

For **Alzheimer**

`brownie run .\scripts\setup.py main --network fl-local` 

#### Setup after first time
`brownie run .\scripts\setup.py --network fl-local`

The number of devices is choosen by the constants

### Step 2: run the collaborators
Open n terminals, where n is the number of nodes of the simulation. In the first n-1 terminals, run the collaborators. Hospital names can be found in _hospital_split.json_.

`brownie run .\scripts\node.py [hospital-name] --network fl-local`

**Notes**: In this configuration you need to wait 3600 s to validate if a device send the weights or not, it possible to change the time to wait, changing the constants TIMEOUT_SECONDS and TIMEOUT_DEVICES for simulation purpose.

### Step 3: run the aggregator
On the last open terminal, run the node that will start as aggregator.
Of course, [hospital-name] can't be the name of an hospital already running as a collaborator.

`brownie run .\scripts\node.py [hospital-name] aggregator --network fl-local `

## Authors
<ul>
<li>Federico Faccioli</li>
<li>Alessandro Tutone</li>
</ul>

## Possible extensions:
It's possible to test this model with some functionalities seen in the previous version (https://github.com/LorenzoCassano/Blockchain-FederatedLearning):
<ul>
<li>Implement Out of Battery mode</li>
<li>Test functioning using FedProx</li>
<li>Test functioning using brain tumor dataset</li>
</ul>

Also interesting additional features could be:
<ul>
<li>Handling unforeseen aggregator failure using reportTimeout() and reportAggregatorFailure()</li>

</ul>

