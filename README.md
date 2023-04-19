# Deep Q-learning implementation in solving MDP problem

This report discusses deep Q-learning and its implementation in solving a states-partially-observable problem, “Lunar Lander”. In a lot of real-life simulating problems the full state is not available to the agent, and the generalization of the state will be very necessary. This report discusses the function approximation using an artificial neural network in reinforcement learning, which is known as deep neural network. 


There are two main parts of code. One is the lunar lander problem solving and plotting code, it's saved directly under this folder. Another one is the analysis of 4 hyperparameters, which are saved in the folder "Hyperparameter", file name is related to its function.


## Lunar Lander

Lunar Lander is a sophisticated problem in reinforcement learning, the agent should be trained to know how to land on the landing pad properly. Unlike the learning problems in previous assignments which are discrete MDP, the Lunar Lander has continuous states, which are represented by 8 variables:

- (x, y, x˙, y˙, θ, θ˙, legL, legR)

The state variables x, y are the horizontal and vertical position, and x˙, y˙ are the horizontal and vertical speed. θ, θ˙ are the angle and angular speed of the lander. legL, legR are the binary values to show whether the left or right leg of the lander is touching the ground.

To solve this problem, the simple Q-table would be insufficient. After reading the papers, I implemented the deep Q-network (DQN), which is a combination of reinforcement learning and artificial neural network, to estimate the optimal Q function in continuous state space. The Q function will be represented by an artificial neural network, whose weights are trained to produce the optimal Q value for each action.

## Q-Learning and Function Approximation 

In MDP Q-learning, the approximation value of the actions in each state is stored in the Q-table. However, for continuous states, we need function approximation to generalize the value-state function. The parameterized functional form is using a weight vector , and we can write  for a states  and given weights . The function  can be a linear function or a nonlinear function like the polynomial function. But it also can be a function that is produced by an artificial neural network, in which the weights will be the weight for each node in the network.

To get the function approximation right, the function should update the function weight by shifting its prediction value to the target value. Let us refer to an individual update by the notation , where  is the state updated and  is the update target that 's estimated value is shifted toward.In reinforcement learning, the function update should happen while the agent is interacting with the environment, that’s why it is critical to have an algorithm that can efficiently learn from the incrementing data. Also, it is worth mentioning that the target function may not be stationary. The target function will change over time, so the function approximation should also shift to different targets each time.


## Code usage

To run the code, you can open .ipynb with any notebook or run the .py file. I have installed the Jupyter extension in my VScode, not sure will that help with the running or not.

Link to the Jupyter extension for VScode: 
>	https://marketplace.visualstudio.com/items?itemName=donjayamanne.jupyter



RUN THE CODE:
> 1. Run the code blocks one by one and you will get all the graphs I show in the report
> 2. The first line of each code block has a comment explaining what the code block does
