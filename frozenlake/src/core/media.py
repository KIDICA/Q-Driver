import pyglet


class AudioPlayer:
    """Wrapper for pyglet media API."""

    def __init__(self, file):
        self.player = pyglet.media.Player()
        self.player.queue(pyglet.media.load(file))
        self.playing = False

    def play(self, loop=False):
        if self.playing:
            return

        self.player.loop = loop
        self.player.play()
        self.playing = True

    def stop(self):
        self.player.stop()
        self.playing = False

    def pause(self):
        self.player.pause()
        self.playing = False
