import pandas as pd
from typing import List, Tuple

from extractors.SceneData import SceneData
from extractors.config import *
from extractors.TrackDirection import TrackDirection


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
        """_summary_

        Args:
            trackDf (pd.DataFrame): NORTH is positive y, EAST is positive x

        Returns:
            Tuple[TrackDirection]: NORTH/SOUTH, EAST/WEST
        """
        # if local y is decreasing, then SOUTH
        # if local x is increasing, then EAST
        verticalDirection = TrackDirection.NORTH
        horizontalDirection = TrackDirection.EAST
        firstRow = trackDf.head(1).iloc[0]
        lastRow = trackDf.tail(1).iloc[0]
        if firstRow[self.positionCols[1]] > lastRow[self.positionCols[1]]:
            verticalDirection = TrackDirection.SOUTH

        if firstRow[self.positionCols[0]] > lastRow[self.positionCols[0]]:
            horizontalDirection = TrackDirection.WEST

        return verticalDirection, horizontalDirection

    def breakScenePedTrajectoriesInto3Parts(self, sceneData: SceneData, midOffset: float = 1.5):
        """returns 3 dataframes: start, mid, finish

        Args:
            sceneData (SceneData): _description_
            midOffset (float): mid section starts and ends by midOffset from the boundary of the scene bounding box.
        """

        pedDf = sceneData.getClippedPedDfs()

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

    def getTrajectoriesInDirection(self, df: pd.DataFrame, direction="north") -> pd.DataFrame:

        trackIds = df[self.idCol].unique()
        for trackId in trackIds:
            trackDf = df[df[self.idCol] == trackId]

    def getAVelocitySeries(self, aPedDf: pd.DataFrame, onCol, fps):
        seriesVelo = aPedDf[onCol].rolling(window=2).apply(
            lambda values: (values.iloc[0] - values.iloc[1]) / (1 / fps))
        seriesVelo.iloc[0] = seriesVelo.iloc[1]
        return seriesVelo
    
    def getVelocitySeries(self, pedDf: pd.DataFrame, onCol, fps):
        pedVelocities = []
        for pedId in pedDf["uniqueTrackId"].unique():
            aPed = pedDf[pedDf["uniqueTrackId"]==pedId] 
            pedVelocities.append(self.getAVelocitySeries(aPed, onCol, fps))

        velSeries = pd.concat(pedVelocities)
        return velSeries

