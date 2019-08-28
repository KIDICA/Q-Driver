import arcade
import os
from custom_frozenlake import FrozenLake
import numpy as np
import threading
import time

lake = FrozenLake()
SIZE = lake.size
SPRITE_SIZE = 128
SCREEN_TITLE = 'Sprite Bouncing Coins'

image = {
    'UNKNOWN': 'images/unknown.png',
    'CHARACTER': 'images/character.png',
    'EMPTY': 'images/empty2.png',
    'GROUND': 'images/ground.png',
    'BOX': 'images/box.png',
    'GOAL': 'images/coin.png'
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
        super().__init__(width, width, 'FrozenLake')

        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        self.q_learner = qlearner

        # self.coin_list = None
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

        self.MOVEMENT_SPEED = 5
        self.is_running = True
        self.key_down = False

        # Transpose the map entries, the draw-order makes it necessary.
        self.map = np.array(list(map(lambda row: list(row), self.lake.MAP))).transpose()

        self.uncovered = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        self.uncovered[0][0] = 1
        self.player = self.sprite_from_index(0, 0, image['CHARACTER'])
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

    def act(self, move, state, action_prob, steps, rewards):
        self.rewards += rewards
        self.info2 = f"Q(s={state}, a')=" + ', '.join(list(map(lambda x: f"{x:.5f}", action_prob)))
        if move == LEFT:
            self.left()
        elif move == RIGHT:
            self.right()
        elif move == UP:
            self.up()
        elif move == DOWN:
            self.down()
        return self.is_running

    def start(self):
        self.setup()
        if self.q_learner is not None:
            self.thread = threading.Thread(target=lambda: self.q_learner.train(lambda move, state, action_prob, steps, rewards: self.act(move, state, action_prob, steps, rewards)))
            self.thread.daemon = True
            self.thread.start()  # Start the execution
        arcade.run()
        # q.save()

    def sprite(self, x, y, img='images/box.png'):
        s = arcade.Sprite(img, self.SPRITE_SCALING)
        s.center_x = x
        s.center_y = y
        return s

    def on_close(self):
        self.is_running = False
        if self.q_learner is not None:
            self.thread.do_run = False
            self.thread.join()

    def coord(self, x, y):
        """Turns grid coordinates into drawing coordinates."""
        return x * self.SPRITE_RESIZE + 3 * self.SPRITE_RESIZE_HALF, self.SCREEN_HEIGHT - y * self.SPRITE_RESIZE - self.SPRITE_RESIZE - self.SPRITE_RESIZE_HALF

    def sprite_from_index(self, x, y, img):
        _x, _y = self.coord(x, y)
        return self.sprite(_x, _y, img)

    def setup(self):
        self.wall_list = arcade.SpriteList()
        self.sprites = arcade.SpriteList()

        # Level borders
        for x in range(self.SIZE_WITH_EDGE):
            offset = self.SPRITE_RESIZE * x + self.SPRITE_RESIZE_HALF
            # Bottom edge
            self.wall_list.append(self.sprite(x=offset, y=self.SCREEN_HEIGHT - self.SPRITE_RESIZE_HALF))
            # Top edge
            self.wall_list.append(self.sprite(x=offset, y=self.SPRITE_RESIZE_HALF))
            # Left, x = Distance from left wall
            self.wall_list.append(self.sprite(y=offset, x=self.SPRITE_RESIZE_HALF))
            # Right
            self.wall_list.append(self.sprite(y=offset, x=self.SCREEN_WIDTH - self.SPRITE_RESIZE_HALF))

        for x in range(self.SIZE):
            for y in range(self.SIZE):
                c = self.map[x][y]
                s = None
                if c == GOAL:
                    s = self.sprite_from_index(x, y, image['GOAL'])
                elif c == HOLE:
                    s = self.sprite_from_index(x, y, image['EMPTY'])

                if s is not None:
                    self.sprites.append(s)

        arcade.set_background_color(arcade.color.EARTH_YELLOW)

    def on_draw(self):
        # This command has to happen before we start drawing
        arcade.start_render()

        self.wall_list.draw()
        self.sprites.draw()
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if self.uncovered[x][y] == 0:
                    self.sprite_from_index(x, y, image['UNKNOWN']).draw()
        self.player.draw()
        arcade.draw_text(f"Score: {self.score} Deaths: {self.deaths} Steps: {self.move_count} {self.info}", self.SPRITE_RESIZE_HALF, self.SCREEN_HEIGHT - self.SPRITE_RESIZE_HALF - 20, arcade.color.WHITE, font_size=30, bold=True)
        arcade.draw_text(f"Moves: {' > '.join(self.moves)}", self.SPRITE_RESIZE, self.SPRITE_RESIZE_HALF, arcade.color.WHITE, font_size=25, bold=True)
        arcade.draw_text(self.info2, self.SPRITE_RESIZE, self.SPRITE_RESIZE_HALF / 4, arcade.color.WHITE, font_size=25, bold=True)
        if self.message is not None:
            arcade.draw_text(self.message, self.SCREEN_WIDTH / 2.5, self.SCREEN_HEIGHT / 2, arcade.color.WHITE, font_size=50, bold=True)

    def up(self):
        self.move_count += 1
        self.player_y = max(0, self.player_y - 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(UP)

    def down(self):
        self.move_count += 1
        self.player_y = min(self.SIZE - 1, self.player_y + 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(DOWN)

    def left(self):
        self.move_count += 1
        self.player_x = max(0, self.player_x - 1)
        self.uncovered[self.player_x][self.player_y] = 1
        self.push_move(LEFT)

    def right(self):
        self.move_count += 1
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
        if key == arcade.key.DOWN:
            self.down()
        if key == arcade.key.RIGHT:
            self.right()
        if key == arcade.key.LEFT:
            self.left()

        self.key_down = True

    def on_key_release(self, symbol: int, modifiers: int):
        self.key_down = False

    def dead(self):
        self.player_x = 0
        self.player_y = 0
        self.score -= 10
        self.deaths += 1

    def won(self):
        self.has_won = True
        self.message = 'WON!!!'

    def restart(self):
        time.sleep(2)
        self.t.do_run = False
        self.t.join()
        self.reset()

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

    def update(self, delta_time):
        c = self.map[self.player_x][self.player_y]

        if c == HOLE:
            self.dead()
        elif c == GOAL:
            self.won()
            self.t = threading.Thread(target=self.restart)
            self.t.start()

        self.player.center_x, self.player.center_y = self.coord(self.player_x, self.player_y)
