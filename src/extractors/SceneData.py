import pandas as pd
import numpy as np
from shapely.geometry import Point
from tools.TrajectoryUtils import TrajectoryUtils
from loguru import logger
from tqdm import tqdm
from .config import *
import os
from dill import dump, load



class SceneData:
    """SceneData only has crossing trajectories.
    """

    def __init__(
        self,
        locationId,
        orthoPxToMeter,
        sceneId,
        sceneConfig,
        boxWidth,
        boxHeight,
        pedData: pd.DataFrame,
        otherData: pd.DataFrame,
        backgroundImagePath=None
    ):
        self.locationId = locationId
        self.orthoPxToMeter = orthoPxToMeter  # for visualization
        self.sceneId = sceneId
        self.sceneConfig = sceneConfig
        self.centerX = sceneConfig["centerX"]
        self.centerY = sceneConfig["centerY"]
        self.angle = sceneConfig["angle"]
        self.backgroundImagePath = backgroundImagePath

        self.boxWidth = boxWidth
        self.boxHeight = boxHeight
        self.polygon = TrajectoryUtils.scenePolygon(
            sceneConfig, boxWidth, boxHeight)

        self.pedData = pedData
        self._clippedPedData = None
        self._pedDataLocal = None
        self._pedIds = None

        self.otherData = otherData
        self._clippedOtherData = None
        self._otherDataLocal = None
        self._otherIds = None

        self._dropWorldCoordinateColumns()
        self._transformToLocalCoordinates()

    def uniquePedIds(self) -> np.ndarray:
        if self._pedIds is None:
            self._pedIds = self.pedData.uniqueTrackId.unique()

        return self._pedIds

    def uniqueClippedPedIds(self) -> np.ndarray:
        clippedDf = self.getClippedPedDfs()
        if len(clippedDf) > 0:
            return clippedDf.uniqueTrackId.unique()
        return []

    def uniqueOtherIds(self) -> np.ndarray:
        if self._otherIds is None:
            self._otherIds = self.otherData.uniqueTrackId.unique()

        return self._otherIds

    def uniqueClippedOtherIds(self) -> np.ndarray:
        clippedDf = self.getClippedOtherDfs()
        if len(clippedDf) > 0:
            return clippedDf.uniqueTrackId.unique()
        return []

    def getClippedPedDfByUniqueTrackId(self, uniqueTrackId):
        return self.getPedDfByUniqueTrackId(uniqueTrackId, True)

    def getPedDfByUniqueTrackId(self, uniqueTrackId, clipped=False):
        return self.getPedDfByUniqueTrackIds([uniqueTrackId], clipped=clipped)

    def getPedDfByUniqueTrackIds(self, uniqueTrackIds, clipped=False):

        if clipped:
            clippedDf = self.getClippedPedDfs()
            criterion = clippedDf['uniqueTrackId'].map(
                lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
            return clippedDf[criterion]
        else:
            criterion = self.pedData['uniqueTrackId'].map(
                lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
            return self.pedData[criterion]

    def getOtherDfByUniqueTrackId(self, uniqueTrackId, clipped=False):
        return self.getOtherDfByUniqueTrackIds([uniqueTrackId], clipped=clipped)

    def getOtherDfByUniqueTrackIds(self, uniqueTrackIds, clipped=False):

        if clipped:
            clippedDf = self.getClippedOtherDfs()
            criterion = clippedDf['uniqueTrackId'].map(
                lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
            return clippedDf[criterion]
        else:
            criterion = self.otherData['uniqueTrackId'].map(
                lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
            return self.otherData[criterion]

    def _dropWorldCoordinateColumns(self):
        logger.debug(
            "Dropping , lonVelocity, latVelocity, lonAcceleration, latAcceleration")
        self.pedData = self.pedData.drop(
            ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1)
        self.otherData = self.otherData.drop(
            ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1)

    def transformToLocalCoordinate(self):
        logger.debug("transforming trajectories to scene coordinates")

        # translate and rotate.
        pedDf = self.getClippedPedDfs()
        self._pedDataLocal = self._transformDfToLocalCoordinates(pedDf)

        otherDf = self.getClippedOtherDfs()
        self._otherDataLocal = self._transformDfToLocalCoordinates(otherDf)

        pass

    def _transformDfToLocalCoordinates(self, df):

        # transform position
        # transform velocity
        # transform heading

        origin = Point(self.centerX, self.centerY)
        originAngle = self.angle

        translationMat = TrajectoryUtils.getTranslationMatrix(origin)
        rotationMat = TrajectoryUtils.getRotationMatrix(originAngle)

        sceneX = []
        sceneY = []

        for idx, row in df.iterrows():

            position = Point(row["xCenter"], row["yCenter"])
            # velocity = (row["xVelocity"], row["yVelocity"])
            # acceleration = (row["xAcceleration"], row["yAcceleration"])
            # heading = row['heading']
            newPosition = TrajectoryUtils.transformPoint(
                translationMat, rotationMat, position)

            # row['sceneX'] = newPosition.x
            # row['sceneY'] = newPosition.y
            sceneX.append(newPosition.x)
            sceneY.append(newPosition.y)

        df["sceneX"] = sceneX
        df["sceneY"] = sceneY
        return df

    def getPedDataInSceneCorrdinates(self):
        if self._pedDataLocal is None:
            self.transformToLocalCoordinate()

        return self._pedDataLocal

    def getOtherDataInSceneCorrdinates(self):
        if self._otherDataLocal is None:
            self.transformToLocalCoordinate()

        return self._otherDataLocal

    # region clipping

    def _clipPed(self):
        logger.debug("clipping trajectories")
        scenePolygon = TrajectoryUtils.scenePolygon(
            self.sceneConfig, self.sceneConfig["boxWidth"], self.sceneConfig["roadWidth"] + CLIP_OFFSET)
        dfs = []
        for pedId in tqdm(self.uniquePedIds(), desc=f"clipping ped trajectories for scene # {self.sceneId}"):
            pedDf = self.getPedDfByUniqueTrackId(pedId)
            clippedDf = TrajectoryUtils.clipByRect(
                pedDf, "xCenter", "yCenter", "frame", scenePolygon)
            if TrajectoryUtils.length(clippedDf, "xCenter", "yCenter") < self.sceneConfig["roadWidth"]:
                logger.debug(
                    f"Disregarding trajectory for {pedId} because the length is too low")
            else:
                dfs.append(clippedDf)

        if len(dfs) == 0:
            """No pedData"""
            self._clippedPedData = pd.DataFrame()
        else:
            self._clippedPedData = pd.concat(dfs, ignore_index=True)

    def _clipOther(self):
        logger.debug("clipping other trajectories")
        # we will clip with the bounding box + 50 meters
        scenePolygon = TrajectoryUtils.scenePolygon(
            self.sceneConfig, self.sceneConfig["boxWidth"] + OTHER_CLIP_LENGTH, self.sceneConfig["roadWidth"])
        dfs = []
        for otherId in tqdm(self.uniqueOtherIds(), desc=f"clipping other trajectories for scene # {self.sceneId}"):
            otherDf = self.getOtherDfByUniqueTrackId(otherId)
            clippedDf = TrajectoryUtils.clipByRect(
                otherDf, "xCenter", "yCenter", "frame", scenePolygon)
            if TrajectoryUtils.length(clippedDf, "xCenter", "yCenter") < self.sceneConfig["roadWidth"]:
                logger.debug(
                    f"Disregarding trajectory for {otherId} because the length is too low")
            else:
                dfs.append(clippedDf)

        if len(dfs) == 0:
            """No pedData"""
            self._clippedOtherData = pd.DataFrame()
        else:
            self._clippedOtherData = pd.concat(dfs, ignore_index=True)

    def getClippedPedDfs(self):
        if self._clippedPedData is None:
            self._clipPed()

        return self._clippedPedData

    def getClippedOtherDfs(self):
        if self._clippedOtherData is None:
            self._clipOther()

        return self._clippedOtherData

    def clippedPedSize(self):
        return len(self.uniqueClippedPedIds())

    def clippedOtherSize(self):
        return len(self.uniqueClippedOtherIds())


    def saveDataframes(self, pathPrefix:str):
        """
            saves clipped dataframes only
        """
        fpath = f"{pathPrefix}-scene-{self.sceneId}-pedestrians.csv"
        if os.path.exists(fpath):
            os.remove(fpath)
        
        self.getPedDataInSceneCorrdinates().to_csv(fpath, index=False)
        # self.getClippedPedDfs().to_csv(fpath, index=False)

        fpath = f"{pathPrefix}-scene-{self.sceneId}-others.csv"
        if os.path.exists(fpath):
            os.remove(fpath)
        
        self.getOtherDataInSceneCorrdinates().to_csv(fpath, index=False)
        # self.getClippedOtherDfs().to_csv(fpath, index=False)