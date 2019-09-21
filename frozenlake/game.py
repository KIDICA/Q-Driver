import arcade
import os
import numpy as np
import threading
import time
from buttons import Button
from collections import deque
from custom_frozenlake import FrozenLake
import argparse

lake = FrozenLake()
SIZE = lake.size
SPRITE_SIZE = 128
SCREEN_TITLE = 'FrozenLake Q-Learning'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--pack', dest='pack', type=str, default='asphalt')
parser.add_argument('-p', dest='pack', type=str, default='asphalt')
parser.add_argument('--ground-edges', dest='ground_edges', action='store_true', default=False)
parser.add_argument('--draw-freq', dest='draw_freq', action='store_true', default=False)
args = parser.parse_args()

pack = args.pack
GROUND_EDGES = args.ground_edges
DRAW_FREQ = args.draw_freq

# Assets
image = {
    'UNKNOWN': f"images/{pack}/hidden.png",
    'CHAR_LEFT': f'images/{pack}/char_left.png',
    'CHAR_RIGHT': f'images/{pack}/char_right.png',
    'CHAR_UP': f'images/{pack}/char_up.png',
    'CHAR_DOWN': f'images/{pack}/char_down.png',
    'EMPTY': f'images/{pack}/hole.png',
    'GOAL': f'images/{pack}/goal.png',
    'TREE': f'images/{pack}/decor.png',
    'EDGE_TOP': f'images/{pack}/edge_top.png',
    'EDGE_LEFT': f'images/{pack}/edge_left.png',
    'EDGE_RIGHT': f'images/{pack}/edge_right.png',
    'EDGE_BOTTOM': f'images/{pack}/edge_bottom.png',
    'EDGE_BOTTOM_LEFT': f'images/{pack}/edge_bottom_left.png',
    'EDGE_BOTTOM_RIGHT': f'images/{pack}/edge_bottom_right.png',
    'EDGE_TOP_LEFT': f'images/{pack}/edge_top_left.png',
    'EDGE_TOP_RIGHT': f'images/{pack}/edge_top_right.png',
    'GROUND': f'images/{pack}/ground.png',
    'GROUND_TOP_LEFT': f'images/{pack}/ground_top_left.png',
    'GROUND_TOP_RIGHT': f'images/{pack}/ground_top_right.png',
    'GROUND_BOTTOM_LEFT': f'images/{pack}/ground_bottom_left.png',
    'GROUND_BOTTOM_RIGHT': f'images/{pack}/ground_bottom_right.png',
    'GROUND_TOP': f'images/{pack}/ground_top.png',
    'GROUND_LEFT': f'images/{pack}/ground_left.png',
    'GROUND_RIGHT': f'images/{pack}/ground_right.png',
    'GROUND_BOTTOM': f'images/{pack}/ground_bottom.png',
    'KILL': f'images/{pack}/kill.png',
}

sound = {
    'AMBIENT': 'sounds/ambient.wav',
    'HONK': 'sounds/honk.wav'
}


HOLE = 'H'
GROUND = 'F'
START = 'S'
GOAL = 'G'

UP = 'UP'
LEFT = 'LEFT'
RIGHT = 'RIGHT'
DOWN = 'DOWN'


class FrozenLakeGame(arcade.Window):
    def __init__(self, width, height, qlearner=None):
        super().__init__(width + 300, width, SCREEN_TITLE)

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

        self.sound_ambient = arcade.load_sound(sound['AMBIENT'])
        self.sound_honk = arcade.load_sound(sound['HONK'])

        self.is_running = True
        self.key_down = False

        # Transpose the map entries, the draw-order makes it necessary.
        self.map = np.array(list(map(lambda row: list(row), self.lake.map))).transpose()

        self.uncovered = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        self.uncovered[0][0] = 1
        self.player_up = self.sprite_from_index(0, 0, image['CHAR_UP'])
        self.player_down = self.sprite_from_index(0, 0, image['CHAR_DOWN'])
        self.player_left = self.sprite_from_index(0, 0, image['CHAR_LEFT'])
        self.player_right = self.sprite_from_index(0, 0, image['CHAR_RIGHT'])
        self.player = self.player_right
        self.player_x = 0
        self.player_y = 0
        self.set_mouse_visible(True)
        self.score = 0
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
        #self.thread_executor: threading.Thread = None
        self.executing = False
        self.episode_q = None
        self.episode_current_state = 0
        self.episode_last_state = None

        self.episodes = 0
        self.episode = 0
        self.steps = 0

        self.selection = None

        self.text_offset_y = self.SPRITE_RESIZE
        self.text_offset_x = self.SPRITE_RESIZE
        self.text_font_size = 15
        self.text_line_spacing = 20

        self.button_list = []
        self.memory = deque()

        self.button_list.append(
            Button(self.SCREEN_WIDTH + 80, self.SCREEN_HEIGHT - 100, '+ε', lambda: self.q_learner.inc_epsilon()))
        self.button_list.append(
            Button(self.SCREEN_WIDTH + 130, self.SCREEN_HEIGHT - 100, '-ε', lambda: self.q_learner.dec_epsilon()))

        self.button_list.append(
            Button(self.SCREEN_WIDTH + 80, self.SCREEN_HEIGHT - 250, '+α', lambda: self.q_learner.inc_alpha()))
        self.button_list.append(
            Button(self.SCREEN_WIDTH + 130, self.SCREEN_HEIGHT - 250, '-α', lambda: self.q_learner.dec_alpha()))

        self.button_list.append(
            Button(self.SCREEN_WIDTH + 80, self.SCREEN_HEIGHT - 390, '+γ', lambda: self.q_learner.inc_gamma()))
        self.button_list.append(
            Button(self.SCREEN_WIDTH + 130, self.SCREEN_HEIGHT - 390, '-v', lambda: self.q_learner.dec_gamma()))

        self.button_list.append(
            Button(self.SCREEN_WIDTH + 80, self.SCREEN_HEIGHT - 550, '+t', lambda: self.q_learner.inc_updates()))
        self.button_list.append(
            Button(self.SCREEN_WIDTH + 130, self.SCREEN_HEIGHT - 550, '-t', lambda: self.q_learner.dec_updates()))

    def inc_updates(self):
        self.updates_per_second += 1

    def dec_updates(self):
        self.updates_per_second = max(1, self.updates_per_second - 1)

    def has_beaten_game(self, steps):
        if steps % 5000 == 0:
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

    def act(self, action, steps, reward, episode, episodes):
        self.rewards = reward
        self.episode = episode + 1
        self.episodes = episodes
        self.steps = steps
        # self.rewards += rewards
        # self.info2 = f"Q(s={state}, a')=" + ', '.join(list(map(lambda x: f"{x:.5f}", action_prob)))
        if action == LEFT:
            self.left()
        elif action == RIGHT:
            self.right()
        elif action == UP:
            self.up()
        elif action == DOWN:
            self.down()

        # if (self.has_beaten_game(steps)):
        #    print('Game beaten, done...')

        return self.is_running

    def start(self):
        self.setup()
        self.spawn()
        self.sound_ambient.play()
        # the engine doesn't surface the underlying API for looping, this just works when I hacked the core lib.
        #self.sound_ambient.play(loop=True)
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
                if action == UP:
                    self.up()
                elif action == LEFT:
                    self.left()
                elif action == RIGHT:
                    self.right()
                elif action == DOWN:
                    self.down()

            time.sleep(1 / self.updates_per_second)

    def spawn(self):
        """The ML algorithm needs to run within another thread, otherwise it will block the game engine loop."""
        if self.q_learner is not None:
            self.thread = threading.Thread(
                target=lambda: self.q_learner.train(
                    lambda action, steps, reward, episode, episodes: self.act(action, steps, reward, episode,
                                                                              episodes)))
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
            return None
        if y < self.SPRITE_RESIZE or y > (self.SCREEN_HEIGHT - self.SPRITE_RESIZE):
            return None

        grid_x, grid_y = x // self.SPRITE_RESIZE - 1, self.SIZE - y // self.SPRITE_RESIZE

        return grid_x, grid_y, self.map[grid_x][grid_y]

    def sprite_from_index(self, x, y, img):
        _x, _y = self.coord(x, y)
        return self.sprite(_x, _y, img)

    def setup(self):
        self.border_ground = arcade.SpriteList()
        self.wall_list = arcade.SpriteList()
        self.sprites = arcade.SpriteList()

        # Level borders
        LAST = self.SIZE_WITH_EDGE - 1
        for x in range(self.SIZE_WITH_EDGE):
            for y in range(self.SIZE_WITH_EDGE):
                pos_x = self.SPRITE_RESIZE * x + self.SPRITE_RESIZE_HALF
                pos_y = self.SPRITE_RESIZE * y + self.SPRITE_RESIZE_HALF
                img = None
                if x == 0 and y == 0:
                    img = image['EDGE_BOTTOM_LEFT']
                elif x == LAST and y == 0:
                    img = image['EDGE_BOTTOM_RIGHT']
                elif x == 0 and y == LAST:
                    img = image['EDGE_TOP_LEFT']
                elif x == LAST and y == LAST:
                    img = image['EDGE_TOP_RIGHT']
                elif y == 0 and 0 < x < LAST:
                    img = image['EDGE_BOTTOM']
                elif y == LAST and 0 < x < LAST:
                    img = image['EDGE_TOP']
                elif x == 0 and 0 < y < LAST:
                    img = image['EDGE_LEFT']
                elif x == LAST and 0 < y < LAST:
                    img = image['EDGE_RIGHT']

                if img is not None:
                    self.border_ground.append(self.sprite(x=pos_x, y=pos_y, img=img))

                # Bottom edge
                # self.border_ground.append(self.sprite(x=offset, y=self.SCREEN_HEIGHT - self.SPRITE_RESIZE_HALF, img=img))
                # self.wall_list.append(self.sprite(x=offset, y=self.SCREEN_HEIGHT - self.SPRITE_RESIZE_HALF))
                # Top edge
                # self.border_ground.append(self.sprite(x=offset, y=self.SPRITE_RESIZE_HALF, img=img))
                # self.wall_list.append(self.sprite(x=offset, y=self.SPRITE_RESIZE_HALF))
                # Left, x = Distance from left wall
                # self.border_ground.append(self.sprite(y=offset, x=self.SPRITE_RESIZE_HALF, img=img))
                # self.wall_list.append(self.sprite(y=offset, x=self.SPRITE_RESIZE_HALF))
                # Right
                # self.border_ground.append(self.sprite(y=offset, x=self.SCREEN_WIDTH - self.SPRITE_RESIZE_HALF, img=img))
                # self.wall_list.append(self.sprite(y=offset, x=self.SCREEN_WIDTH - self.SPRITE_RESIZE_HALF))

        # Inner field
        LAST = self.SIZE - 1
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                img = image['GROUND']
                # The edges of the inner field can also have special tiles.
                if GROUND_EDGES:
                    if x == 0 and y == 0:
                        img = image['GROUND_TOP_LEFT']
                    elif x == LAST and y == 0:
                        img = image['GROUND_TOP_RIGHT']
                    elif x == 0 and y == LAST:
                        img = image['GROUND_BOTTOM_LEFT']
                    elif x == LAST and y == LAST:
                        img = image['GROUND_BOTTOM_RIGHT']
                    elif y == 0 and 0 < x < LAST:
                        img = image['GROUND_TOP']
                    elif y == LAST and 0 < x < LAST:
                        img = image['GROUND_BOTTOM']
                    elif x == 0 and 0 < y < LAST:
                        img = image['GROUND_LEFT']
                    elif x == LAST and 0 < y < LAST:
                        img = image['GROUND_RIGHT']
                s = self.sprite_from_index(x, y, img)
                self.sprites.append(s)

        arcade.set_background_color(arcade.color.CHARCOAL)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.selection is not None:
            x, y, _ = self.cart_to_grid(x, y)
            self.map[x][y] = self.selection

        for button in self.button_list:
            button.check_click(x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        for button in self.button_list:
            if button.pressed:
                button.on_release()

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
                c = self.map[x][y]
                s = None
                if c == GOAL:
                    s = self.sprite_from_index(x, y, image['GOAL'])
                elif c == HOLE:
                    s = self.sprite_from_index(x, y, image['EMPTY'])
                # elif c == GROUND:
                #    s = self.sprite_from_index(x, y, image['GROUND'])
                if s is not None:
                    s.draw()

                if self.uncovered[x][y] == 0:
                    self.sprite_from_index(x, y, image['UNKNOWN']).draw()
                elif self.map[x][y] == HOLE:
                    self.sprite_from_index(x, y, image['KILL']).draw()

        self.player.draw()

        # Draw the buttons
        for button in self.button_list:
            button.draw()

        arcade.draw_text(f"ε: {round(self.q_learner.EPSILON, 2)}", self.SCREEN_WIDTH + 30, self.SCREEN_HEIGHT - 50,
                         arcade.color.WHITE, font_size=25, bold=True)
        arcade.draw_text(f"α: {round(self.q_learner.LEARNING_RATE, 2)}", self.SCREEN_WIDTH + 30,
                         self.SCREEN_HEIGHT - 200,
                         arcade.color.WHITE, font_size=25, bold=True)
        arcade.draw_text(f"γ: {round(self.q_learner.DISCOUNT_FACTOR, 2)}", self.SCREEN_WIDTH + 30,
                         self.SCREEN_HEIGHT - 350,
                         arcade.color.WHITE, font_size=25, bold=True)
        arcade.draw_text(f"1/t: {self.q_learner.updates_per_second}", self.SCREEN_WIDTH + 30, self.SCREEN_HEIGHT - 500,
                         arcade.color.WHITE, font_size=25, bold=True)
        arcade.draw_text(f"steps: {self.steps}", self.SCREEN_WIDTH + 20, 100,
                         arcade.color.WHITE, font_size=15, bold=True)
        arcade.draw_text(f"deaths: {self.deaths}", self.SCREEN_WIDTH + 20, 75,
                         arcade.color.WHITE, font_size=15, bold=True)
        arcade.draw_text(f"episode: {self.episode}/{self.episodes}", self.SCREEN_WIDTH + 20, 50,
                         arcade.color.WHITE, font_size=15, bold=True)

        if DRAW_FREQ:
            self.draw_freq()

        # texts = self.hud_text()
        # for i, text in enumerate(texts):
        #   arcade.draw_text(text, self.text_offset_x, self.text_line_spacing * i + self.text_offset_y,
        #                     arcade.color.BLACK, font_size=self.text_font_size, bold=True)

    def hud_text(self):
        messages = [
            f"Score: {self.score} Deaths: {self.deaths} Steps: {self.move_count} {self.info}",
            f"Moves: {' > '.join(self.moves)}",
            self.info2
        ]
        if self.message is not None:
            messages.append(self.message)

        return messages

    def up(self):
        self.move_count += 1
        self.player = self.player_up
        self.player_y = max(0, self.player_y - 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(UP)

    def down(self):
        self.move_count += 1
        self.player = self.player_down
        self.player_y = min(self.SIZE - 1, self.player_y + 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(DOWN)

    def left(self):
        self.move_count += 1
        self.player = self.player_left
        self.player_x = max(0, self.player_x - 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(LEFT)

    def right(self):
        self.move_count += 1
        self.player = self.player_right
        self.player_x = min(self.SIZE - 1, self.player_x + 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(RIGHT)

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
            self.selection = HOLE
        elif key == arcade.key.G:
            self.selection = GROUND

        self.key_down = True

    def on_key_release(self, symbol: int, modifiers: int):
        self.key_down = False

    def dead(self):
        self.player_x = 0
        self.player_y = 0
        self.score -= 10
        self.deaths += 1
        arcade.play_sound(self.sound_honk)

    def won(self):
        self.has_won = True
        self.message = 'WON!!!'

    def restart(self):
        time.sleep(2)
        self.t.do_run = False
        self.t.join()
        self.reset()

    def respawn(self):
        self.player_x = 0
        self.player_y = 0

    def reset(self):
        """The entire game is reset here."""
        self.player_x = 0
        self.player_y = 0
        self.score = 0
        self.deaths = 0
        self.move_count = 0
        self.has_won = False
        self.message = None
        self.uncovered = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        self.uncovered[0][0] = 1
        self.is_running = True
        self.rewards = 0
        self.selection = None

    def update(self, delta_time):
        c = self.map[self.player_x][self.player_y]

        if c == HOLE:
            self.dead()
        elif c == GOAL:
            self.won()
            self.t = threading.Thread(target=self.restart)
            self.t.start()

        self.player.center_x, self.player.center_y = self.coord(self.player_x, self.player_y)


if __name__ == "__main__":
    from custom_frozenlake import FrozenLake
    from q_learn import QLearner
    from screeninfo import get_monitors

    height = int(get_monitors()[0].height * 0.85)
    game = FrozenLakeGame(height, 900, QLearner(FrozenLake()))
    game.start()
