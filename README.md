# Q-Driver

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
python src/game.py
```

The simulation supports texture-packs listed in the `images` folder:

```
python src/game.py -p forest
```

## Simulation

The simulation allows adjusting three parameters of the value-iteration function with buttons on the right. All values range from [0, 1].

1. __Epsilon__ is the _exploration rate_, the higher the move random paths are crossed.
1. __Alpha__ (also known as _learning rate_) defines how much old values are preferred (1.0) over new values (0.0) that are learned across the way.
1. __Gamma__ (also known as _discount factor_) reduces the value of future rewards.
   1. This has two-fold reasons:
      1. Shorter paths are preferred over long ones.
      1. Solution will converge (to 0), which also prevents to walk infinitely in circles.

### Different build in texture-packs

You might want to customize the assets for your scenario, just create a new folder and keep the naming conventions of the existing files.

![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_asphalt.png)
![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_dirt.png)
![](https://raw.githubusercontent.com/KIDICA/ai-hacking/master/doc/img/frozen_forest.png)

### Video

---

[![](https://i9.ytimg.com/vi/c5AKOSVi0pQ/mq3.jpg?sqp=CNSskuwF&rs=AOn4CLBw6U_2kP8eLVcJYJw69SWH3AAYXg)](https://www.youtube.com/watch?v=c5AKOSVi0pQ)


### Heatmap

---

The engine has built in support to generate a heatmaps to show how many steps has been spent in each state relative to each other.

The state with value of 1 has been frequented the most, all other state have lower values.
A value of 0 should indicate that the state has not been occupied at all (game rules).

This should also give an idea how inefficient RL often is and how random the approximation was (based on the hyper parameters).

[![](https://i9.ytimg.com/vi/5OIhS4n9Kfw/mq3.jpg?sqp=CIarkuwF&rs=AOn4CLB5MTMZTtduGqxUvOvJcZg7DmAHDQ)](https://www.youtube.com/watch?v=5OIhS4n9Kfw)

## References

### Credits

Assets are taken from:

1. https://www.kenney.nl/assets
1. https://opengameart.org/content/lpc-terrain-repack-2
1. https://opengameart.org/content/lpc-sandrock-alt-colors
1. https://opengameart.org/content/lpc-modified-art
1. http://soundbible.com/royalty-free-sounds-1.html

### Resources

The implementations are based on research from multiple source, some of those are:

1. [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands)\
Maxim Lapan June 20, 2018
1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)\
Richard S. Sutton and Andrew G. Barto, Second Edition, MIT Press, Cambridge, MA, 2018
1. [iX 05/2019, iX 06/2019, iX 07/2019](https://shop.heise.de/katalog/ix-05-2019-pdf)\
Three part series on RL (German).
1. https://geektutu.com (Chinese)

## License

MIT
