import gymgym_custom_frozenlake
import matplotlib.pyplot as plt
import _thread

env = gym.make("MsPacman-v0")
actions = env.get_action_meanings()

env.reset()
img = plt.imshow(env.render(mode='rgb_array'))


def input_thread(a_list):
    input()
    a_list.append(True)


def start():
    done = False
    i = 1
    a_list = []
    _thread.start_new_thread(input_thread, (a_list,))
    while not a_list and not done and (i < 100):
        img.set_data(env.render(mode='rgb_array'))
        plt.draw()
        plt.pause(0.05)
        a = env.action_space.sample()
        s, r, done, info = env.step(a)
        print(i, ":", "action:", actions[a], ", rewards:", r, ", done:", done)
        i += 1


start()
