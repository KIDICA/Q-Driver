import os

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_files(path):
    """Returns all files within a folder."""
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


class DynamicImport:
    """
    This breaks compile time benefits and editor type-checking and auto-completion.
    Can be used but is certainly not convenient and debugging friendly.
    """

    def __init__(self, pack, asset_type):
        self.pack = pack
        self.asset_type = asset_type

        path = f"{MODULE_PATH}/../../pack/{self.pack}/textures"
        files = get_files(path)
        for file in files:
            filename = os.path.splitext(file)[0]
            key, value = str.upper(filename), self._path(file)
            setattr(self, key, value)

    def _path(self, file):
        return f"{MODULE_PATH}/../../pack/{self.pack}/{self.asset_type}/{file}"


class Texture:
    def __init__(self, pack):
        self.pack = pack
        self.UNKNOWN = self._path('hidden.png')
        self.CHAR_LEFT = self._path('char_left.png')
        self.CHAR_RIGHT = self._path('char_right.png')
        self.CHAR_UP = self._path('char_up.png')
        self.CHAR_DOWN = self._path('char_down.png')
        self.EMPTY = self._path('hole.png')
        self.HOLE = self._path('hole.png')
        self.GOAL = self._path('goal.png')
        self.TREE = self._path('decor.png')
        self.EDGE_TOP = self._path('edge_top.png')
        self.EDGE_LEFT = self._path('edge_left.png')
        self.EDGE_RIGHT = self._path('edge_right.png')
        self.EDGE_BOTTOM = self._path('edge_bottom.png')
        self.EDGE_BOTTOM_LEFT = self._path('edge_bottom_left.png')
        self.EDGE_BOTTOM_RIGHT = self._path('edge_bottom_right.png')
        self.EDGE_TOP_LEFT = self._path('edge_top_left.png')
        self.EDGE_TOP_RIGHT = self._path('edge_top_right.png')
        self.GROUND = self._path('ground.png')
        self.GROUND_TOP_LEFT = self._path('ground_top_left.png')
        self.GROUND_TOP_RIGHT = self._path('ground_top_right.png')
        self.GROUND_BOTTOM_LEFT = self._path('ground_bottom_left.png')
        self.GROUND_BOTTOM_RIGHT = self._path('ground_bottom_right.png')
        self.GROUND_TOP = self._path('ground_top.png')
        self.GROUND_LEFT = self._path('ground_left.png')
        self.GROUND_RIGHT = self._path('ground_right.png')
        self.GROUND_BOTTOM = self._path('ground_bottom.png')
        self.KILL = self._path('kill.png')

    def _path(self, file):
        return f"{MODULE_PATH}/../../pack/{self.pack}/textures/{file}"


class Sound:
    def __init__(self, pack):
        self.pack = pack
        self.AMBIENT = self._path('ambient.wav')
        self.HONK = self._path('honk.wav')

    def _path(self, file):
        return f"{MODULE_PATH}/../../pack/{self.pack}/sounds/{file}"
