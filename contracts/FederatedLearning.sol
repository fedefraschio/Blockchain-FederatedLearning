// SPDX-License-Identifier: MIT

pragma solidity ^0.8.28;
pragma experimental ABIEncoderV2;

import "./chainlink/AggregatorV3Interface.sol";
// Import OpenZeppelin's AccessControl contract
import "@openzeppelin/contracts/access/AccessControl.sol";

contract FederatedLearning is AccessControl {
    //using SafeMathChainlink for uint256; // l'overflow e l'underflow aritmetico sono gestiti automaticamente
    // Define a constant role for the aggregator
    bytes32 public constant AGGREGATOR_ROLE = keccak256("AGGREGATOR_ROLE"); // NEW
    
    // Enumeration of federated learning states
    enum FL_STATE {
        CLOSE,
        OPEN,
        START,
        LEARNING
    }

    // State variables
    FL_STATE public fl_state;              // Current state of the federated learning system
    address[] public collaborators;        // List of approved collaborators
    address public aggregator;             // Address of the aggregator
    bytes public model;                    // Serialized model data
    bytes public compile_info;             // Metadata about the model
    bytes public aggregated_weights;       // Aggregated weights after learning

    mapping(address => bytes) public weights;          // Stores individual collaborator weights
    uint256 public weights_len;                        // Count of submitted weights
    uint256 public roundTimeout;                       // Timeout for each learning round
    uint256 private lastElectedIndex = 0;              // Tracks the last elected collaborator's index

    mapping(address => mapping(string => bool)) public hasCalledFunction; // Tracks if a collaborator has called a specific function
    mapping(string => uint) public everyoneHasCalled;  // Tracks how many collaborators have called a specific function

    // Timeout-related variables
    mapping(address => bool) public hasReportedTimeout; // Tracks timeout reports per collaborator
    uint256 public timeoutReportCount;                  // Number of timeout reports received
    uint256 public timeoutReportThreshold;              // Threshold of timeout reports required to take action
    uint256 public roundStartTime;                      // Timestamp when the current round started

    // Events to notify system state changes and actions
    event StartState();
    event LearningState();
    event CloseState();
    event EveryCollaboratorHasCalledOnlyOnce(string functionName);
    event AggregatedWeightsReady();
    event RoundProceeded(); // NEW
    event TimeoutReported(address reporter); // NEW
    event NewAggregatorElected(); // NEW

    // Constructor to initialize the contract
    constructor(uint256 _roundTimeout, uint256 _timeoutReportThreshold) {
        fl_state = FL_STATE.CLOSE; // Set initial state to CLOSE
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender); // Set admin role to th\e contract deployer
        roundTimeout = _roundTimeout; // Set the round timeout duration
        timeoutReportThreshold = _timeoutReportThreshold; // Set the threshold for timeout reports
    }

    // Modifier to restrict access to authorized users
    modifier onlyAuthorized() {
        require(isAuthorized(msg.sender), "Unauthorized user");
        _;
    }
    
    // Modifier to restrict access to the aggregator
    modifier onlyAggregator() {
        require(isAggregator(msg.sender), "Non-aggregator user");
        _;
    }
    
    // Modifier to ensure each collaborator calls the function only once
    modifier everyCollaboratorHasCalledOnce(string memory functionName) {
        require(
            !hasCalledFunction[msg.sender][functionName],
            "This function can only be called only once per collaborator"
        );
        hasCalledFunction[msg.sender][functionName] = true; // Mark the function as called for this collaborator

        everyoneHasCalled[functionName]++;
        if (everyoneHasCalled[functionName] == (collaborators.length)-1) {
            emit EveryCollaboratorHasCalledOnlyOnce(functionName);
        }

        _;
    }
    
    // Function to check if a user is authorized
    function isAuthorized(address _user) public view returns (bool) {
        if (hasRole(DEFAULT_ADMIN_ROLE, _user)) {
            return true;
        }
        for (uint i = 0; i < collaborators.length; i++) {
            if (collaborators[i] == _user) {
                return true;
            }
        }
        return false;
    }

    // Function to check if a user is the aggregator
    function isAggregator(address _user) public view returns (bool) {
        if (hasRole(DEFAULT_ADMIN_ROLE, _user)) {
            return true;
        }
        if (aggregator == _user) {
                return true;
        }
        return false;
    }

    // Open the system for collaborators
    function open() public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(fl_state == FL_STATE.CLOSE || fl_state == FL_STATE.OPEN);
        fl_state = FL_STATE.OPEN;
    }

    // Add a new collaborator
    function add_collaborator(address _collaborator) public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(fl_state == FL_STATE.OPEN);
        collaborators.push(_collaborator);
    }

    // Set the initial model
    function send_model(bytes memory _model) public onlyAggregator() {
        require(fl_state == FL_STATE.OPEN);
        model = _model;
    }

    // Provide compilation metadata
    function send_compile_info(bytes memory _compile_info) public onlyAggregator() {
        require(fl_state == FL_STATE.OPEN);
        compile_info = _compile_info;
    }

    // Transition to the START state
    function start() public onlyAggregator() {
        require(fl_state == FL_STATE.OPEN);
        fl_state = FL_STATE.START;
        emit StartState();
    }

    // Retrieve the model
    function retrieve_model()
        public
        onlyAuthorized
        everyCollaboratorHasCalledOnce("retrieve_model")
        returns (bytes memory)
    {
        require(fl_state == FL_STATE.START, "Not in START state");
        return model;
    }

    // Function to retrieve compilation information
    function retrieve_compile_info()
        public
        onlyAuthorized
        everyCollaboratorHasCalledOnce("retrieve_compile_info")
        returns (bytes memory)
    {
        require(fl_state == FL_STATE.START, "Not in START state");
        return compile_info;
    }

    // Function to transition to the LEARNING state
    function learning() public onlyAggregator {
        require(fl_state == FL_STATE.START, "Not in START state");
        fl_state = FL_STATE.LEARNING;
        emit LearningState();
    }

    // Function for collaborators to send their weights
    function send_weights(
        bytes memory _weights
    ) public onlyAuthorized everyCollaboratorHasCalledOnce("send_weights") {
        require(fl_state == FL_STATE.LEARNING, "Not in LEARNING state");
        weights_len++;
        require(weights_len <= collaborators.length);
        weights[msg.sender] = _weights;
    }

    // Function to retrieve the weights submitted by a specific collaborator
    function retrieve_weights(
        address _collaborator
    ) public view onlyAggregator returns (bytes memory) {
        require(fl_state == FL_STATE.LEARNING, "Not in LEARNING state");
        return weights[_collaborator];
    }

    // Function to reset weights for a new learning round
    function reset_weights() public onlyAggregator {
        for (uint256 i = 0; i < collaborators.length; i++) {
            address collaborator = collaborators[i];
            delete hasCalledFunction[collaborator]["send_weights"];
        }
        delete everyoneHasCalled["send_weights"];
    }

    // Function to send aggregated weights after processing
    function send_aggregated_weights(bytes memory _weights) public onlyAggregator {
        require(fl_state == FL_STATE.LEARNING);
        aggregated_weights = _weights;
        weights_len = 0;

        for (uint256 i = 0; i < collaborators.length; i++) {
            delete weights[collaborators[i]];
        }

        emit AggregatedWeightsReady();
    }

    // Function to retrieve the aggregated weights
    function retrieve_aggregated_weights()
        public
        view
        onlyAuthorized
        returns (bytes memory)
    {
        require(fl_state == FL_STATE.LEARNING, "Not in LEARNING state");
        return aggregated_weights;
    }

    // Function to reset the aggregated weights for a new round
    function reset_aggregated_weights() public onlyAggregator {
        for (uint256 i = 0; i < collaborators.length; i++) {
            address collaborator = collaborators[i];
            delete hasCalledFunction[collaborator][
                "retrieve_aggregated_weights"
            ];
        }
        delete everyoneHasCalled["retrieve_aggregated_weights"];
    }

    // Function to reset all state variables when closing the contract
    function resetContractState() internal {
        // Resetting call state for each collaborator
        for (uint i = 0; i < collaborators.length; i++) {
            hasCalledFunction[collaborators[i]]["retrieve_model"] = false;
            hasCalledFunction[collaborators[i]]["send_weights"] = false;
            hasCalledFunction[collaborators[i]]["aggregated_weights"] = false;
            hasCalledFunction[collaborators[i]]["retrieve_compile_info"] = false;
            hasReportedTimeout[collaborators[i]] = false;
        }

        // Resetting call number
        everyoneHasCalled["retrieve_model"] = 0;
        everyoneHasCalled["send_weights"] = 0;
        everyoneHasCalled["aggregated_weights"] = 0;
        everyoneHasCalled["retrieve_compile_info"] = 0;
        timeoutReportCount = 0;

        // Resetting state variables
        aggregated_weights = "";
        model = "";
        compile_info = "";
        weights_len = 0;
        roundStartTime = 0; 
        roundTimeout = 0;

        // Deleting saved weigths
        for (uint i = 0; i < collaborators.length; i++) {
            delete weights[collaborators[i]];
        }

        // Resetting aggregator address
        aggregator = address(0);

        // Keeping CLOSE state
        fl_state = FL_STATE.CLOSE;
    }


    // Function to transition to the CLOSE state
    function close() public onlyRole(DEFAULT_ADMIN_ROLE) {
        fl_state = FL_STATE.CLOSE;
        resetContractState();
        emit CloseState();
    }


    // Function to get the current state as a string
    function get_state() public view returns (string memory) {
        if (fl_state == FL_STATE.CLOSE) return "CLOSE";
        if (fl_state == FL_STATE.OPEN) return "OPEN";
        if (fl_state == FL_STATE.START) return "START";
        if (fl_state == FL_STATE.LEARNING) return "LEARNING";
        return "No State";
    }

    // Function to get the list of all collaborators
    function get_collaborators() public view returns (address[] memory) {
        return collaborators;
    }

    // Function to get the model data
    function get_model() public view onlyAuthorized returns (bytes memory) {
        return model;
    }

    // Function to get the compilation information
    function get_compile_info()
        public
        view
        onlyAuthorized
        returns (bytes memory)
    {
        return compile_info;
    }

    // Function to get the aggregated weights
    function get_aggregated_weights()
        public
        view
        onlyAuthorized
        returns (bytes memory)
    {
        return aggregated_weights;
    }

    // Collaborative timeout handling
    function reportTimeout() public onlyAuthorized {
        require(fl_state == FL_STATE.LEARNING, "Not in learning state");
        require(block.timestamp >= roundStartTime + roundTimeout,
                    "Timeout not reached");
        require(!hasReportedTimeout[msg.sender], "Already reported timeout");
        hasReportedTimeout[msg.sender] = true;
        timeoutReportCount++;
        emit TimeoutReported(msg.sender);
        // Proceed only if enough reports received
        if (timeoutReportCount >= timeoutReportThreshold)
           emit RoundProceeded();
    }

    // Function to handle aggregator failure reports
    function reportAggregatorFailure() public onlyAuthorized {
        require(fl_state == FL_STATE.LEARNING, "Not in learning state");
        require(block.timestamp >= roundStartTime + roundTimeout, "Timeout not reached");
        require(!hasReportedTimeout[msg.sender], "Already reported timeout");
    
        hasReportedTimeout[msg.sender] = true;
        timeoutReportCount++;

        emit TimeoutReported(msg.sender);

        // If the number of reports exceeds the threshold, emit a failure event or take action
        if (timeoutReportCount >= timeoutReportThreshold) {
            // Reset state or reassign aggregator role if necessary
            fl_state = FL_STATE.CLOSE; // Example: Transition to a safe state
            emit CloseState(); // Notify system of state change
        }
    }

    // Function to elect a new aggregator in a round-robin manner
    function electNewAggregator() public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(fl_state == FL_STATE.CLOSE, "Can only elect a new aggregator when the state is CLOSE");
        require(collaborators.length > 0, "No collaborators available to elect as aggregator");

        // Revoke previous aggregator role
        if (aggregator != address(0)) {
            revokeRole(AGGREGATOR_ROLE, aggregator);
        }
        
        // Round-robin selection logic
        aggregator = collaborators[lastElectedIndex];

        // Grant the new aggregator the AGGREGATOR_ROLE
        grantRole(AGGREGATOR_ROLE, aggregator);

        // Update the index to the next collaborator, wrapping around if necessary
        lastElectedIndex = (lastElectedIndex + 1) % collaborators.length;

        emit NewAggregatorElected();
    }

    // Function to get the new aggregator
    function get_aggregator() public view returns (address) {
        return aggregator;
    }

}
