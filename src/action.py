from enum import Enum
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    @classmethod
    def get_all_actions(cls):
        return [Action.UP , Action.DOWN , Action.LEFT , Action.RIGHT]