# AI-Hacking

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

## Simulation

### Different build in texture-packs

The simulation allows adjusting three parameters of the value-iteration function with buttons on the right. All values range from [0, 1].

1. __Epsilon__ is the exploration rate, the higher the move random paths are crossed.
1. __Alpha__ (also known as _learning rate_) defines how much old values are preferred (1.0) over new values (0.0) that are learned across the way.
1. __Gamma__ (also known as discount factor) reduces the value of future rewards.
   1. This has two-fold reasons:
      1. Shorter paths are preferred over long ones.
      1. Iteration converges to 0 an will finally terminate, which also prevents to walk infinitely in circles.

You might want to customize the assets for your scenario, just create a new folder and keep the naming conventions of the existing files.

![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_asphalt.png)
![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_dirt.png)
![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_forest.png)

### Video

[![](https://i9.ytimg.com/vi/c5AKOSVi0pQ/mq3.jpg?sqp=CNSskuwF&rs=AOn4CLBw6U_2kP8eLVcJYJw69SWH3AAYXg)](https://www.youtube.com/watch?v=c5AKOSVi0pQ)


#### Heat-map

The engine has built in support to generate a heat-map of the states showing how many steps has been spent in each cell in proportion to each other, the state with most steps has a value of 1 and any other cell has a value smaller 1.
This show the inefficient the exploration is.

[![](https://i9.ytimg.com/vi/5OIhS4n9Kfw/mq3.jpg?sqp=CIarkuwF&rs=AOn4CLB5MTMZTtduGqxUvOvJcZg7DmAHDQ)](https://www.youtube.com/watch?v=5OIhS4n9Kfw)

## Reference

### Assets

1. Game graphics are mainly taken from: https://www.kenney.nl/assets

1. Sound: http://soundbible.com/royalty-free-sounds-1.html

### Resources

The implementations are based on research from multiple source, some of those are:

1. [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands)
Maxim Lapan June 20, 2018

1. Richard S. Sutton and Andrew G. Barto, Second Edition, MIT Press, Cambridge, MA, 2018
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)

1. [iX Developer Machine Learning](https://shop.heise.de/katalog/ix-developer-machine-learning)
This journal has quite some breaking bugs in the demo code, so be aware.

1. https://geektutu.com (Chinese)

## License

MIT