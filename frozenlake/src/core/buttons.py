import arcade


class Button:
    """Base button type."""

    def __init__(self, center_x, center_y, width, height, button_height=2, on_click=None):
        self.center_x: int = center_x
        self.center_y: input() = center_y
        self.width: int = width
        self.height: int = height
        self.pressed: bool = False
        self.button_height = button_height
        self.on_click = on_click

    def draw(self):
        pass

    def on_press(self):
        if self.pressed:
            return

        self.pressed = True
        if self.on_click is not None:
            self.on_click()

    def on_release(self):
        self.pressed = False

    def _in_range(self, x: int, y: int):
        if x > self.center_x + self.width // 2:
            return False
        if x < self.center_x - self.width // 2:
            return False
        if y > self.center_y + self.height // 2:
            return False
        if y < self.center_y - self.height // 2:
            return False

        return True

    def click(self, x: int, y: int):
        if self.pressed:
            return

        if self._in_range(x, y):
            self.on_press()

    def unclick(self, x: int, y: int):
        if self._in_range(x, y):
            self.on_release()


class Buttons:
    def __init__(self, button_list=[]):
        self.buttons: list[Button] = button_list

    def draw(self):
        for button in self.buttons:
            button.draw()

    def click(self, x: int, y: int):
        for button in self.buttons:
            button.click(x, y)

    def unclick(self, x, y):
        for button in self.buttons:
            button.unclick(x, y)


class TextButton(Button):
    def __init__(self, center_x, center_y, text, font_size=18, font_face="Arial",
                 face_color=arcade.color.ANTIQUE_WHITE,
                 highlight_color=arcade.color.WHITE,
                 shadow_color=arcade.color.ASH_GREY,
                 on_click=None):
        super().__init__(center_x=center_x, center_y=center_y, width=40, height=40, on_click=on_click)
        self.text = text
        self.font_size = font_size
        self.font_face = font_face
        self.face_color = face_color
        self.highlight_color = highlight_color
        self.shadow_color = shadow_color

    def draw(self):
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width,
                                     self.height, self.face_color)

        if not self.pressed:
            color = self.shadow_color
        else:
            color = self.highlight_color

        # Bottom horizontal
        arcade.draw_line(self.center_x - self.width / 2, self.center_y - self.height / 2,
                         self.center_x + self.width / 2, self.center_y - self.height / 2,
                         color, self.button_height)

        # Right vertical
        arcade.draw_line(self.center_x + self.width / 2, self.center_y - self.height / 2,
                         self.center_x + self.width / 2, self.center_y + self.height / 2,
                         color, self.button_height)

        if not self.pressed:
            color = self.highlight_color
        else:
            color = self.shadow_color

        # Top horizontal
        arcade.draw_line(self.center_x - self.width / 2, self.center_y + self.height / 2,
                         self.center_x + self.width / 2, self.center_y + self.height / 2,
                         color, self.button_height)

        # Left vertical
        arcade.draw_line(self.center_x - self.width / 2, self.center_y - self.height / 2,
                         self.center_x - self.width / 2, self.center_y + self.height / 2,
                         color, self.button_height)

        x = self.center_x
        y = self.center_y
        if not self.pressed:
            x -= self.button_height
            y += self.button_height

        arcade.draw_text(self.text, x, y,
                         arcade.color.BLACK, font_size=self.font_size,
                         width=self.width, align="center",
                         anchor_x="center", anchor_y="center")


class ImageButton(Button):
    """
    Text-based button
    Based on: http://arcade.academy/examples/gui_text_button.html
     """

    def __init__(self, center_x, center_y, width, height, img, scale=1, on_click=None):
        super().__init__(center_x=center_x, center_y=center_y, width=40, height=40, on_click=on_click)
        self.sprite = arcade.Sprite(filename=img, image_width=width, image_height=height, center_x=center_x, center_y=center_y, scale=scale)

    def draw(self):
        self.sprite.draw()

        if not self.pressed:
            color = self.shadow_color
        else:
            color = self.highlight_color

        # Bottom horizontal
        arcade.draw_line(self.center_x - self.width / 2, self.center_y - self.height / 2,
                         self.center_x + self.width / 2, self.center_y - self.height / 2,
                         color)

        # Right vertical
        arcade.draw_line(self.center_x + self.width / 2, self.center_y - self.height / 2,
                         self.center_x + self.width / 2, self.center_y + self.height / 2,
                         color)

        if not self.pressed:
            color = self.highlight_color
        else:
            color = self.shadow_color

        # Top horizontal
        arcade.draw_line(self.center_x - self.width / 2, self.center_y + self.height / 2,
                         self.center_x + self.width / 2, self.center_y + self.height / 2,
                         color)

        # Left vertical
        arcade.draw_line(self.center_x - self.width / 2, self.center_y - self.height / 2,
                         self.center_x - self.width / 2, self.center_y + self.height / 2,
                         color)

        x = self.center_x
        y = self.center_y
        if not self.pressed:
            x -= self.button_height
            y += self.button_height
