# AI Hacking

A collection of AI experiments mainly based on Reinforcement Learning methods.

Main part of the project is currently an interactive demo of a grid based Markov Decision Process based on the gym framework's game FrozenLake (but it's loosely insprired by that at this points).

The latest app is a interactive simulation demonstrating Q-Learning with value-iteration with epsilon-greedy.

## Installation

You need to install python 3.7 first.

```
git clone https://github.com/KIDICA/ai-hacking.git
cd ai-hacking
```

Install venv and required packages. Every subfolder has its own venv, because they use vastly different frameworks and implementations:

```
$ cd frozenlake
python3.7 -m pip install venv
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the simulation:

```
python game.py
```

The simulation supports texture-packs listed in the `images` folder:

```
python game.py -p forest
```

## Case

### Different build in texture-packs

You might want to customize the assets for your scenario, just create a new folder and keep the naming conventions of the existing files.

![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_asphalt.png)
![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_dirt.png)
![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_forest.png)

### Video

![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/video/frozen_asphalt.mp4)

#### Heat-map

The engine has built in support to generate a heat-map of the states showing how many steps has been spent in each cell in proportion to each other, the state with most steps has a value of 1 and any other cell has a value smaller 1.
This show the inefficient the exploration is by

![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/video/q_learn_epis_50000_snap_2000_eps_0.9_gamma_0.95_alpha_0.8.mp4)

## Reference

### Assets

Game graphics are mainly taken from: https://www.kenney.nl/assets

Sound: http://soundbible.com/royalty-free-sounds-1.html

### Resources

The implementations are based on multiple source:

[Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands)

Richard S. Sutton and Andrew G. Barto, Second Edition, MIT Press, Cambridge, MA, 2018
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)

[iX Developer Machine Learning](https://shop.heise.de/katalog/ix-developer-machine-learning)
This journal has quite some breaking bugs in the demo code, so be aware.

https://geektutu.com (Chinese)

## License

MIT