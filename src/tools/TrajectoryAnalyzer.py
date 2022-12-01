import pandas as pd
from typing import List, Tuple

from extractors.SceneData import SceneData
from extractors.config import *
from extractors.TrackDirection import TrackDirection
from .TrajectoryUtils import TrajectoryUtils


class TrajectoryAnalyzer:

    def __init__(
        self,
        fps,
        idCol,
        positionCols,
        velocityCols,
        accelerationCols
    ):

        self.fps = fps
        self.idCol = idCol
        self.positionCols = positionCols
        self.velocityCols = velocityCols
        self.accelerationCols = accelerationCols

        pass

    def getTrack_VH_Directions(self, trackDf: pd.DataFrame) -> Tuple[TrackDirection]:
        return TrajectoryUtils.getTrack_VH_Directions(trackDf, self.positionCols[0], self.positionCols[1])


    def breakScenePedTrajectoriesInto3Parts(self, sceneData: SceneData, midOffset: float = 1.5):
        """returns 3 dataframes: start, mid, finish

        Args:
            sceneData (SceneData): _description_
            midOffset (float): mid section starts and ends by midOffset from the boundary of the scene bounding box.
        """

        pedDf = sceneData.getPedDataInSceneCoordinates()
        sceneMeta = sceneData.getMeta()
        NSpedIds = sceneMeta

        # we get trajectories with y-axis
        # Each pedestrian can start near y = 0 or y = maxY
        maxY = sceneData.sceneConfig["roadWidth"] / 2
        minY = -maxY

        # TODO incorrect, this can sometimes have peds' finishing section

        

        firstPart = self.clipByYaxis(pedDf, minY=minY, maxY=minY + midOffset)
        midPart = self.clipByYaxis(
            pedDf, minY=minY + midOffset, maxY=maxY - midOffset)
        lastPart = self.clipByYaxis(pedDf, minY=maxY - midOffset, maxY=maxY)

    def clipByYaxis(self, df: pd.DataFrame, minY: float, maxY: float) -> pd.DataFrame:
        filter = (df[self.positionCols[1]] >= minY) & (
            df[self.positionCols[1]] <= maxY)

        return df[filter].copy()

    # def getTrajectoriesInDirection(self, df: pd.DataFrame, direction="north") -> pd.DataFrame:

    #     trackIds = df[self.idCol].unique()
    #     for trackId in trackIds:
    #         trackDf = df[df[self.idCol] == trackId]


