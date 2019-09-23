import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from ml.q_learn import QLearner


class Drawer:
    def __init__(self, q: QLearner):
        # Norm to 1.0
        print('steps', q.steps)
        q.state_visits = q.state_visits / q.max_visit
        q.state_visits[-1][-1][-1] = -1
        self.q = q
        self.size = len(q.state_visits)
        sns.set()
        # Generate heat map.
        self.fig = plt.figure()
        self.cmap = 'magma'

    def init(self):
        # values = self.q.state_visits[-1]
        # print(values)
        # size = int(math.sqrt(self.q.mdp.state_count()))
        # labels = np.array([[round(values[j][i],3) for i in range(size)] for j in range(size)])
        # print('labels', labels)
        ax = sns.heatmap(self.q.state_visits[-1], cmap=self.cmap, xticklabels=False, yticklabels=False, square=True,
                         annot=True, vmin=0.0, vfmax=1.0, annot_kws={"size": 6}, linewidths=1, cbar=True,
                         linecolor='white')
        ax.set_title(
            f"Value-Iteration: ε={round(self.q.EPSILON, 2)}, γ={round(self.q.DISCOUNT_FACTOR, 2)}, α={round(self.q.LEARNING_RATE, 2)}, steps={self.q.steps}")

    def animate(self, i):
        print("i:", i, self.size)
        data = self.q.state_visits[i]
        sns.heatmap(data, cmap=self.cmap, xticklabels=False, yticklabels=False, square=True, annot=False, vmin=0.0,
                    vmax=1.0, linewidths=1, cbar=False, linecolor='white')

    def save(self):
        fps = 15
        intervals = int(1 / fps * 1000)
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, cache_frame_data=True,
                                       frames=self.size, repeat=False, interval=intervals)
        anim.save(
            f"q_learn_epis_{self.q.EPISODES}_snap_{self.q.SNAPSHOT_STATES}_eps_{self.q.EPSILON}_gamma_{self.q.DISCOUNT_FACTOR}_alpha_{self.q.LEARNING_RATE}.mp4",
            writer='ffmpeg', bitrate=5000, dpi=600)
