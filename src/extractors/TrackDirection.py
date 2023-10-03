from enum import Enum

class TrackDirection(Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    @staticmethod
    def createByValue(value: str):
        if value == "NORTH":
            return TrackDirection.NORTH
        if value == "SOUTH":
            return TrackDirection.SOUTH
        if value == "EAST":
            return TrackDirection.EAST 
        if value == "WEST":
            return TrackDirection.WEST