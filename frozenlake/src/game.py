import arcade
import os
import numpy as np
import threading
import time
from core.buttons import TextButton, ImageButton, Buttons
from collections import deque
from env.frozenlake2 import CustomFrozenLakeEnv as FrozenLake
import argparse
from core.media import AudioPlayer
from core.assets import Sound, Texture

parser = argparse.ArgumentParser(description='Customize the game launcher with following arguments.')
parser.add_argument('--pack', dest='pack', type=str, default='asphalt')
parser.add_argument('-p', dest='pack', type=str, default='asphalt')
parser.add_argument('--ground-edges', dest='ground_edges', action='store_true', default=False)
parser.add_argument('--draw-freq', dest='draw_freq', action='store_true', default=False)
args = parser.parse_args()

# Module vars

PACK = args.pack
GROUND_EDGES = args.ground_edges
DRAW_FREQ = args.draw_freq

lake = FrozenLake()

SIZE = lake.size
SPRITE_SIZE = 128
SCREEN_TITLE = 'FrozenLake Q-Learning'

tex = Texture(PACK)
sound = Sound(PACK)


class Field:
    HOLE = b'H'
    GROUND = b'F'
    START = b'S'
    GOAL = b'G'


FieldName = {None: 'None', Field.HOLE: 'HOLE', Field.GROUND: 'GROUND', Field.START: 'START', Field.GOAL: 'GOAL'}


class Dir:
    UP = 'UP'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    DOWN = 'DOWN'


class Game(arcade.Window):
    def __init__(self, width, height, qlearner=None):
        super().__init__(width + 350, width, SCREEN_TITLE)

        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        self.q_learner = qlearner

        # self.coin_list = None
        self.border_ground = None
        self.wall_list = None
        self.s = None
        self.lake = lake

        self.SIZE = SIZE
        self.SIZE_WITH_EDGE = SIZE + 2
        self.SPRITE_SIZE = SPRITE_SIZE
        # +2 because of the edges/rows, top-bottom-left-right
        self.SPRITE_RESIZE = int(width / (SIZE + 2))
        self.SPRITE_SCALING = self.SPRITE_RESIZE / SPRITE_SIZE
        self.SCREEN_WIDTH = width
        self.SCREEN_HEIGHT = width
        self.TILE_SIZE = int(self.SCREEN_HEIGHT / self.SPRITE_RESIZE)
        self.SPRITE_RESIZE_HALF = int(self.SPRITE_RESIZE / 2)

        self.START_X = self.SPRITE_RESIZE_HALF
        self.START_Y = self.SPRITE_RESIZE_HALF
        self.OFFSET = self.SPRITE_RESIZE_HALF

        self.sound_ambient = AudioPlayer(sound.AMBIENT)
        self.sound_honk = arcade.load_sound(sound.HONK)

        self.is_running = True
        self.key_down = False

        # Transpose the map entries, the screen coordinates draw order makes it necessary.
        self.map = self.lake.map.transpose()

        self.uncovered = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        self.uncovered[0][0] = 1
        self.uncovered[-1][-1] = 1
        self.player_up = self.sprite_from_index(0, 0, tex.CHAR_UP)
        self.player_down = self.sprite_from_index(0, 0, tex.CHAR_DOWN)
        self.player_left = self.sprite_from_index(0, 0, tex.CHAR_LEFT)
        self.player_right = self.sprite_from_index(0, 0, tex.CHAR_RIGHT)
        self.player = self.player_right
        self.player_x = 0
        self.player_y = 0
        self.set_mouse_visible(True)
        self.deaths = 0
        self.move_count = 0
        self.sprites = None
        self.info = ''
        self.info2 = ' '
        self.moves = []
        self.has_won = False
        self.message = None
        self.rewards = 0

        self.updates_per_second = 2
        self.thread: threading.Thread = None
        # self.thread_executor: threading.Thread = None
        self.executing = False
        self.episode_q = None
        self.episode_current_state = 0
        self.episode_last_state = None

        self.episodes = 0
        self.episode = 0
        self.steps = 0
        self.episode_steps = 0

        self.selection = None

        self.text_offset_y = self.SPRITE_RESIZE
        self.text_offset_x = self.SPRITE_RESIZE
        self.text_font_size = 15
        self.text_line_spacing = 20

        self.button_list = Buttons([
            # epsilon
            TextButton(center_x=self.SCREEN_WIDTH + 80, center_y=self.SCREEN_HEIGHT - 100, text='+ε', on_click=self.q_learner.inc_epsilon),
            TextButton(center_x=self.SCREEN_WIDTH + 130, center_y=self.SCREEN_HEIGHT - 100, text='-ε', on_click=self.q_learner.dec_epsilon),
            # alpha
            TextButton(center_x=self.SCREEN_WIDTH + 80, center_y=self.SCREEN_HEIGHT - 250, text='+α', on_click=self.q_learner.inc_alpha),
            TextButton(center_x=self.SCREEN_WIDTH + 130, center_y=self.SCREEN_HEIGHT - 250, text='-α', on_click=self.q_learner.dec_alpha),
            # gamma
            TextButton(center_x=self.SCREEN_WIDTH + 80, center_y=self.SCREEN_HEIGHT - 390, text='+γ', on_click=self.q_learner.inc_gamma),
            TextButton(center_x=self.SCREEN_WIDTH + 130, center_y=self.SCREEN_HEIGHT - 390, text='-v', on_click=self.q_learner.dec_gamma),
            # fps
            TextButton(center_x=self.SCREEN_WIDTH + 80, center_y=self.SCREEN_HEIGHT - 550, text='+t', on_click=self.q_learner.inc_updates),
            TextButton(center_x=self.SCREEN_WIDTH + 130, center_y=self.SCREEN_HEIGHT - 550, text='-t', on_click=self.q_learner.dec_updates),
        ])

        def select(field):
            self.selection = field

        self.image_buttons = Buttons([
            ImageButton(center_x=self.SCREEN_WIDTH + 150, center_y=200, width=50, height=50, img=tex.GROUND, on_click=lambda: select(Field.GROUND)),
            ImageButton(center_x=self.SCREEN_WIDTH + 250, center_y=200, width=50, height=50, img=tex.HOLE, on_click=lambda: select(Field.HOLE))
        ])

        self.memory = deque()

    def inc_updates(self):
        self.updates_per_second += 1

    def dec_updates(self):
        self.updates_per_second = max(1, self.updates_per_second - 1)

    def has_beaten_game(self, steps):
        if steps % 5000 == 0:
            global lake
            lake = FrozenLake()
            # report every 5000 steps, test 100 games to get avarage point score for statistics and verify if it is solved
            rew_average = 0.
            for i in range(100):
                s = lake.reset()
                done = False
                while not done:
                    action = np.argmax(self.q_learner.Q[s, :])
                    s, r, done, info = lake.step(action)  # take step using selected action
                    rew_average += r
            rew_average = rew_average / 100
            print('Episode {} avarage reward: {}'.format(steps, rew_average))

            if rew_average > 0.8:
                # FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.
                # Test it on 0.8 so it is not a one-off lucky shot solving it
                print("Frozen lake solved")
                return True

        return False

    def act(self, action, steps, reward, episode, episodes, episode_steps):
        self.rewards = reward
        self.episode = episode + 1
        self.episodes = episodes
        self.steps = steps
        self.episode_steps = episode_steps
        # self.rewards += rewards
        # self.info2 = f"Q(s={state}, a')=" + ', '.join(list(map(lambda x: f"{x:.5f}", action_prob)))
        if action == Dir.LEFT:
            self.left()
        elif action == Dir.RIGHT:
            self.right()
        elif action == Dir.UP:
            self.up()
        elif action == Dir.DOWN:
            self.down()

        # if (self.has_beaten_game(steps)):
        #    print('Game beaten, done...')

        return self.is_running

    def start(self):
        self.setup()
        self.spawn()
        self.sound_ambient.play(loop=True)
        arcade.run()
        # q.save()

    def draw_freq(self):
        """Draws the relative occupation frequency on each state. Illustrates how many steps the algorithm spent in each state to all other states."""
        for i in range(0, self.SIZE ** 2):
            grid_x = i % self.SIZE
            grid_y = i // self.SIZE
            x, y = self.coord(grid_x, grid_y)
            arcade.draw_text(str(round(self.q_learner.state_occupation(i), 2)), x, y, arcade.color.WHITE, font_size=10,
                             bold=True)

    def store_episode(self, episode):
        self.memory.append(episode)
        return True

    def executor(self):
        while True:
            if not self.executing and len(self.memory) > 0:
                self.respawn()
                self.episode_current_state = 0
                self.episode_q, self.episode_last_state = self.memory.popleft()
                self.executing = True

            print(self.episode_current_state, self.episode_last_state)
            if self.episode_current_state == self.episode_last_state:
                self.executing = False

            if self.executing:
                action_number = np.argmax(self.episode_q[self.episode_current_state, :])
                action = self.q_learner.mdp.move[action_number]
                if action == Dir.UP:
                    self.up()
                elif action == Dir.LEFT:
                    self.left()
                elif action == Dir.RIGHT:
                    self.right()
                elif action == Dir.DOWN:
                    self.down()

            time.sleep(1 / self.updates_per_second)

    def spawn(self):
        """The ML algorithm needs to run within another thread, otherwise it will block the game engine loop."""
        if self.q_learner is not None:
            self.thread = threading.Thread(target=lambda: self.q_learner.train(self.act))
            self.thread.daemon = True
            self.thread.start()

            # self.thread_executor = threading.Thread(target=self.executor)
            # self.thread_executor.daemon = True
            # self.thread_executor.start()

    def sprite(self, x, y, img):
        s = arcade.Sprite(img, self.SPRITE_SCALING)
        s.center_x = x
        s.center_y = y
        return s

    def on_close(self):
        self.is_running = False
        self.thread.do_run = False
        self.thread.join()
        # self.thread_executor.join()
        exit(0)

    def coord(self, x, y):
        """Turns grid coordinates into drawing coordinates."""
        return x * self.SPRITE_RESIZE + 3 * self.SPRITE_RESIZE_HALF, self.SCREEN_HEIGHT - y * self.SPRITE_RESIZE - self.SPRITE_RESIZE - self.SPRITE_RESIZE_HALF

    def cart_to_grid(self, x, y):
        """Convert cartesian coordinates to grid coordinates."""
        # Edges don't belong to the grid.
        if x < self.SPRITE_RESIZE or x > (self.SCREEN_WIDTH - self.SPRITE_RESIZE):
            return None, None
        if y < self.SPRITE_RESIZE or y > (self.SCREEN_HEIGHT - self.SPRITE_RESIZE):
            return None, None

        return x // self.SPRITE_RESIZE - 1, self.SIZE - y // self.SPRITE_RESIZE

    def sprite_from_index(self, x, y, img):
        _x, _y = self.coord(x, y)
        return self.sprite(_x, _y, img)

    def setup(self):
        self.border_ground = arcade.SpriteList()
        self.wall_list = arcade.SpriteList()
        self.sprites = arcade.SpriteList()

        # Outer level frame
        last_index = self.SIZE_WITH_EDGE - 1
        for x in range(self.SIZE_WITH_EDGE):
            for y in range(self.SIZE_WITH_EDGE):
                pos_x = self.SPRITE_RESIZE * x + self.SPRITE_RESIZE_HALF
                pos_y = self.SPRITE_RESIZE * y + self.SPRITE_RESIZE_HALF
                img = None
                if x == 0 and y == 0:
                    img = tex.EDGE_BOTTOM_LEFT
                elif x == last_index and y == 0:
                    img = tex.EDGE_BOTTOM_RIGHT
                elif x == 0 and y == last_index:
                    img = tex.EDGE_TOP_LEFT
                elif x == last_index and y == last_index:
                    img = tex.EDGE_TOP_RIGHT
                elif y == 0 and 0 < x < last_index:
                    img = tex.EDGE_BOTTOM
                elif y == last_index and 0 < x < last_index:
                    img = tex.EDGE_TOP
                elif x == 0 and 0 < y < last_index:
                    img = tex.EDGE_LEFT
                elif x == last_index and 0 < y < last_index:
                    img = tex.EDGE_RIGHT

                if img is not None:
                    self.border_ground.append(self.sprite(x=pos_x, y=pos_y, img=img))

        # Inner field
        last_index = self.SIZE - 1
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                img = tex.GROUND
                # The edges of the inner field can also have special tiles.
                if GROUND_EDGES:
                    if x == 0 and y == 0:
                        img = tex.GROUND_TOP_LEFT
                    elif x == last_index and y == 0:
                        img = tex.GROUND_TOP_RIGHT
                    elif x == 0 and y == last_index:
                        img = tex.GROUND_BOTTOM_LEFT
                    elif x == last_index and y == last_index:
                        img = tex.GROUND_BOTTOM_RIGHT
                    elif y == 0 and 0 < x < last_index:
                        img = tex.GROUND_TOP
                    elif y == last_index and 0 < x < last_index:
                        img = tex.GROUND_BOTTOM
                    elif x == 0 and 0 < y < last_index:
                        img = tex.GROUND_LEFT
                    elif x == last_index and 0 < y < last_index:
                        img = tex.GROUND_RIGHT
                s = self.sprite_from_index(x, y, img)
                self.sprites.append(s)

        arcade.set_background_color(arcade.color.CHARCOAL)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.selection is not None:
            grid_x, grid_y = self.cart_to_grid(x, y)
            if grid_x is not None and grid_y is not None:
                self.set_map(grid_x, grid_y, self.selection)

        self.button_list.click(x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.button_list.unclick(x, y)

    def hud_text(self):
        return [
            "KEYS:",
            "+: Speed up",
            "-: Speed down",
            "G: Place 'ground'",
            "H: Place 'hole'",
            " ",
            "INFO:",
            f"Selection: {FieldName[self.selection]}",
            f"Current path length: {self.episode_steps}",
            f"Total steps: {self.steps}",
            f"Total Deaths: {self.deaths}",
            f"Episode: {self.episode}/{self.episodes}",
        ]

    def hyper_param_text(self):
        return [
            f"ε: {round(self.q_learner.EPSILON, 2)}",
            f"α: {round(self.q_learner.LEARNING_RATE, 2)}",
            f"γ: {round(self.q_learner.DISCOUNT_FACTOR, 2)}",
            f"1/t: {self.q_learner.updates_per_second}",
        ]

    @staticmethod
    def draw_text(texts, start_x, start_y=50, offset_y=25, font_size=15):
        if offset_y > 0:
            texts = reversed(texts)

        for i, text in enumerate(texts):
            arcade.draw_text(text, start_x, start_y + i * offset_y, arcade.color.WHITE, font_size=font_size, bold=True)

    def on_draw(self):
        # This comma    nd has to happen before we start drawing
        arcade.start_render()

        self.border_ground.draw()
        self.wall_list.draw()

        # Changed to allow real-time updates.
        self.sprites.draw()

        # Dynamic drawings
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                c = self.get_map(x, y)
                s = None
                if c == Field.GOAL:
                    s = self.sprite_from_index(x, y, tex.GOAL)
                elif c == Field.HOLE:
                    s = self.sprite_from_index(x, y, tex.HOLE)
                # elif c == GROUND:
                #    s = self.sprite_from_index(x, y, texture.GROUND)
                if s is not None:
                    s.draw()

                if self.uncovered[x][y] == 0:
                    self.sprite_from_index(x, y, tex.UNKNOWN).draw()
                elif self.get_map(x, y) == Field.HOLE:
                    self.sprite_from_index(x, y, tex.KILL).draw()

        self.player.draw()
        self.button_list.draw()

        text2 = self.hyper_param_text()
        Game.draw_text(texts=text2, start_x=self.SCREEN_WIDTH + 40, start_y=self.SCREEN_HEIGHT - 50, offset_y=-150, font_size=25)
        Game.draw_text(texts=self.hud_text(), start_x=self.SCREEN_WIDTH + 40)

        if DRAW_FREQ:
            self.draw_freq()

    def up(self):
        self.move_count += 1
        self.player = self.player_up
        self.player_y = max(0, self.player_y - 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(Dir.UP)

    def down(self):
        self.move_count += 1
        self.player = self.player_down
        self.player_y = min(self.SIZE - 1, self.player_y + 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(Dir.DOWN)

    def left(self):
        self.move_count += 1
        self.player = self.player_left
        self.player_x = max(0, self.player_x - 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(Dir.LEFT)

    def right(self):
        self.move_count += 1
        self.player = self.player_right
        self.player_x = min(self.SIZE - 1, self.player_x + 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(Dir.RIGHT)

    def push_move(self, move):
        if len(self.moves) > 3:
            self.moves.pop(0)
        self.moves.append(move)

    def on_key_press(self, key: int, modifiers: int):
        if self.key_down:
            return

        # delegate
        if key == arcade.key.UP:
            self.up()
        elif key == arcade.key.DOWN:
            self.down()
        elif key == arcade.key.RIGHT:
            self.right()
        elif key == arcade.key.LEFT:
            self.left()
        elif key == arcade.key.H:
            self.selection = Field.HOLE
        elif key == arcade.key.G:
            self.selection = Field.GROUND
        elif key == arcade.key.PLUS:
            self.q_learner.inc_updates()
        elif key == args.key.MINUS:
            self.q_learner.dec_updates()

        self.key_down = True

    def on_key_release(self, symbol: int, modifiers: int):
        self.key_down = False

    def dead(self):
        self.player_x = 0
        self.player_y = 0
        self.deaths += 1
        arcade.play_sound(self.sound_honk)

    def won(self):
        self.has_won = True
        self.message = 'WON!!!'

    def restart(self):
        self.reset()

    def respawn(self):
        self.player_x = 0
        self.player_y = 0

    def reset(self):
        """The entire game is reset here."""
        self.player_x = 0
        self.player_y = 0
        self.deaths = 0
        self.move_count = 0
        self.has_won = False
        self.message = None
        self.uncovered = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        self.uncovered[0][0] = 1
        self.uncovered[-1][-1] = 1
        self.is_running = True
        self.rewards = 0
        self.selection = None

    def get_map(self, x, y):
        return self.map[x][y]

    def set_map(self, x, y, c):
        """"Notice that draw order and logical map order are transposed because of screen coordinates."""
        self.map[x][y] = c
        m = self.lake.map.transpose()
        m[x][y] = c
        self.lake.map = m.transpose()

    def update(self, delta_time):
        if self.has_won:
            self.restart()
            return

        c = self.get_map(self.player_x, self.player_y)

        if c == Field.HOLE:
            self.dead()
        elif c == Field.GOAL:
            self.won()

        self.player.center_x, self.player.center_y = self.coord(self.player_x, self.player_y)


if __name__ == "__main__":
    from env.custom_frozenlake import FrozenLake
    from ml.q_learn import QLearner
    from screeninfo import get_monitors

    height = int(get_monitors()[0].height * 0.9)
    game = Game(height, 900, QLearner(FrozenLake()))
    game.start()
