# Enhancing Trading Performance through Deep Q Network Hyper-parameter Tuning

This project aims to enhance a classic trading algorithm by utilizing deep reinforcement learning techniques. By connecting Python and MQL5 through the use of sockets, we have designed a trading algorithm and implemented a deep Q network agent that continuously learns and optimizes the algorithm's hyper-parameters. This combination leads to a trading system that can adapt to changing market conditions and improve its performance over time.

## Table of Contents

1. [Introduction](#introduction)
2. [Classic Trading Algorithm in MQL5](#classic-trading-algorithm-in-mql5)
3. [Reinforcement Learning Agent in Python](#reinforcement-learning-agent-in-python)
4. [Tips for Better Results](#tips-for-better-results)
5. [Directory Explanation](#directory-explanation)
6. [Contributing](#contributing)

## Introduction

The use of sockets allows for real-time communication between Python and MQL5, enabling the visualization of the trading algorithm's results and the deep Q network agent's performance in real-time. This approach represents a significant step forward in the development of advanced trading systems and holds great promise for the future of finance and AI.

## Classic Trading Algorithm in MQL5

The classic trading algorithm is based on the highest and lowest prices of candles. It calculates the highest and lowest of each of the last `n` candles. If the new price crosses the highest, a buy order is executed. If it passes the lowest, a sell order is executed. When an order is executed, the stop-loss is set at the middle of the highest and lowest prices. The number of candles used to calculate the stop-loss and limit prices (`n`) is determined by the RL agent.

The MQL5 code has the responsibility of sending the last-step-state, new-step-state, applied action, and received reward to the Python environment. The state information includes the stop-loss of the contract, the entrance price, the closed price, and the highest and lowest values of the past 5, 10, 20, 40, 80, and 120 candles (dimension = 15). The action refers to the value of `n` chosen (`n` can be 10, 20, or 40). The reward is given if the contract is stopped and represents the profit from the contract.

## Reinforcement Learning Agent in Python

The Reinforcement Learning (RL) agent is implemented using a Deep Q Network with 15 inputs (corresponding to the state information) and 3 outputs (corresponding to the 3 possible actions: setting `n` to 10, 20, or 40).

The "socket_handling" file employs multithreading to facilitate communication with the Python code. It populates the "input_list" which can be emptied when the run code needs to access it. The "model" and "epsilon" (epsilon-greedy) values are set in the "run.py" file.

The "reinforcement_learning" file contains three classes: "DataPrepare", "NeuralNetwork", and "ReinforcementLearningAgent". The network is a multi-layer perceptron (MLP) with four layers. The "ReinforcementLearningAgent" utilizes the neural network to generate Q-values, and actions are taken based on an epsilon-greedy policy, with epsilon decaying over time.

## Tips for Better Results

1. Visualize the rewards and the loss function of the neural network over time to better understand the training process.
2. Be mindful of the calculation of q-values. It is important to keep these values within a reasonable range (e.g., between -10 to 10) as large values can make the network's learning more difficult.
3. Consider using the RMSprop optimizer, as it has been shown to be effective in some articles (although it's not always the best choice).
4. If you are experiencing difficulties, try adjusting the learning rate, the memory size of the Q-learning algorithm, and the epsilon decay schedule. Experimentation is key!

## Directory Explanation
- connection-test : This include a test to connect MQL and python using sockers. The folder has an explained readme.
- rl-agent : this is the main code which implemented a dqn agent working with MQL classic trading algorithm

## Contributing

This project is open-source, and contributions are welcome! If you have any questions, comments, or suggestions for improvements, please reach out to the project maintainer, Reza Karbasi, or submit an issue or pull request on the [GitHub repository](https://github.com/rezakarbasi/RL-agent-trader).
