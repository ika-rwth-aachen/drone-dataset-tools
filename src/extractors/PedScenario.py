from enum import IntEnum
from typing import List
import pandas as pd

class PedScenarioType(IntEnum):
    """Follows the similar conventions as Trajnet with some exceptions
    https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge

    Args:
        IntEnum (_type_): _description_
    """
    Static = 1
    Linear = 2
    Interacting = 3
    NonInteracting = 4
    LeaderFollower = 5
    CollisionAvoidance = 6
    Group = 7
    OnComingVehicle = 8
    OnComingBicyle = 9
    OnComingCar = 10
    OnComingLargeVehicle = 11
    Others = 8


class PedScenario:

    def __init__(
        self,
        sceneId:int,
        scenarioId:int,
        pedId: int,
        recordingId: int,
        start: int,
        end: int,
        fps: float,
        tags: List[PedScenarioType],
        data: pd.DataFrame,
        crossWalkLength: float
    ):
        self.sceneId = sceneId
        self.scenarioId = scenarioId
        self.pedId = pedId
        self.recordingId = recordingId
        self.start = start
        self.end = end
        self.fps = fps
        self.tags = tags
        self.data = data
        self.crossWalkLength = crossWalkLength
    
    
