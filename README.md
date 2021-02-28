# Tool-path-planning-on-3-axis-milling-machine-using-DRL
## Objective
This project attempts to use a classic DRL algorithm, i.e. DQN to finish tool path plnning task on 3-axis milling machine automatically.
## Implementation
* To realize quantitative analysis, the 3d models need to be voxelized and read as 3d-arrays, in which [binvox](https://www.patrickmin.com/binvox/) tools are applied.
* The 3d-arrays which takes geometric information of original models are then used to built up the simulation scenario.
* DQN algorithm is then used to train the agent (cutter) to interact with the environment (workspace on the machine tool) and finish the task (processing of the material).
* The trained strategies are finally tested and evaluated.
## Tips
* To train or test the neural networks in scenarios in different size, some parameters need to be modified in the scripts.
* A simple GUI system may be added in future releases.
## Demonstration
<div align=center>
![Demo1](https://github.com/Maximilian92/T01-Tool-path-planning-on-3-axis-milling-machine-using-DRL/blob/master/image/RL1%20tested%20in%20Te1%20-%20100%25%20-%200.gif)
![Demo2](https://github.com/Maximilian92/T01-Tool-path-planning-on-3-axis-milling-machine-using-DRL/blob/master/image/RL1%20tested%20in%20Te2%20-%20100%25%20-%200.gif)
![Demo3](https://github.com/Maximilian92/T01-Tool-path-planning-on-3-axis-milling-machine-using-DRL/blob/master/image/RL1%20tested%20in%20Te3%20-%20100%25%20-%200.gif)
<br>
<div align=left>
According to the demos, the well trained processing strategies will have good performance as well as generalization ability. 