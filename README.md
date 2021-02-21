# Deep reinforcement learning for guidewire navigation in coronary artery phantom

A demo code for deep reinforcement learning based guidewire navigation in coronary phantom. 
Even if we provide the reinforcement learning model and environment code, it is hard to reproduce this research since the AI agent has been trained and tested using real world robotic manipulator and 2D/3D coronary artery phantoms. To address these limitations, we share preprocessed images of coronary artery phantom, and provide the code that the reinforcement learning algorithm interacts with the dummy environment composed of share images. The reinforcement learning model has been adopted from that of [rl_algorithms](https://github.com/medipixel/rl_algorithms).

## Prerequisites
You need python 3.6 or higher to test this code. We recommend using Anaconda virtual environment using the command below:
```
$ conda create -n rl-pci-demo python=3.6.9
$ conda activate rl-pci-demo
```

## Installation
Install rl_algorithms package and dependencies by using command below:
```
make setup
```

## How to use
Create the pre-trained reinforcement learning model and dummy environment and let them interact each other. Observe what kinds of actions the reinforcement learning model produce which has been trained at a particular state. 

```
python run.py
```