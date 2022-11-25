import pandas as pd
import numpy as np
from shapely.geometry import Point
from tools.TrajectoryUtils import TrajectoryUtils
from .TrackClass import TrackClass
from loguru import logger
from tqdm import tqdm
from .config import *
from .TrackDirection import TrackDirection
import os
from dill import dump, load
from typing import List, Tuple
import logging

# from tools.TrajectoryAnalyzer import TrajectoryAnalyzer


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
        self.fps = FPS
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
        self._pedDataLocal = None # refers to the _clippedPedData, but ensures that local dynamics are added
        self._pedIds = None

        self.otherData = otherData
        self._clippedOtherData = None
        self._otherDataLocal = None # refers to the _clippedOtherData, but ensures that local dynamics are added
        self._otherIds = None
        self._sceneTrackMeta = None

        # self.trajAnalyzer = TrajectoryAnalyzer(
        #     fps=FPS,
        #     idCol="uniqueTrackId",
        #     positionCols=("sceneX", "sceneY"),
        #     velocityCols=("xVelocity", "yVelocity"),
        #     accelerationCols=("xAcceleration", "yAcceleration")
        # )
        print(len(self.pedData))
        self._dropWorldCoordinateColumns()
        self._transformToLocalCoordinates()
        self._addLocalDynamics()
        self._trimHeadAndTailForLocal()
        self._clipPed(crossingOffset = CROSSING_CLIP_OFFSET_AFTER_DYNAMICS) # another pass as we had bigger offset to calculate dynamics
        self._buildSceneTrackMeta()

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

    def filterByIds(self, df: pd.DataFrame, uniqueTrackIds: List[int]) -> pd.DataFrame:
        criterion = df['uniqueTrackId'].map(
            lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
        return df[criterion]


    def getClippedPedDfByUniqueTrackId(self, uniqueTrackId):
        return self.getPedDfByUniqueTrackId(uniqueTrackId, True)

    def getPedDfByUniqueTrackId(self, uniqueTrackId, clipped=False):
        return self.getPedDfByUniqueTrackIds([uniqueTrackId], clipped=clipped)

    def getPedDfByUniqueTrackIds(self, uniqueTrackIds, clipped=False):

        if clipped:
            clippedDf = self.getClippedPedDfs()
            return self.filterByIds(clippedDf, uniqueTrackIds)
        else:
            return self.filterByIds(self.pedData, uniqueTrackIds)

    def getOtherDfByUniqueTrackId(self, uniqueTrackId, clipped=False):
        return self.getOtherDfByUniqueTrackIds([uniqueTrackId], clipped=clipped)

    def getOtherDfByUniqueTrackIds(self, uniqueTrackIds, clipped=False):

        if clipped:
            clippedDf = self.getClippedOtherDfs()
            return self.filterByIds(clippedDf, uniqueTrackIds)
        else:
            return self.filterByIds(self.otherData, uniqueTrackIds)

    def _dropWorldCoordinateColumns(self):
        logger.debug(
            "Dropping , lonVelocity, latVelocity, lonAcceleration, latAcceleration")
        self.pedData = self.pedData.drop(
            ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1)
        self.otherData = self.otherData.drop(
            ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1)

    def _transformToLocalCoordinates(self):
        logging.debug("transforming trajectories to scene coordinates")

        # translate and rotate.
        pedDf = self.getClippedPedDfs()
        self._pedDataLocal = self._transformDfToLocalCoordinates(pedDf)

        otherDf = self.getClippedOtherDfs()
        self._otherDataLocal = self._transformDfToLocalCoordinates(otherDf)

        pass

    def _transformDfToLocalCoordinates(self, df: pd.DataFrame):

        # transform position
        # transform velocity (we cannot do it by translation and rotation)
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

    def _addLocalDynamics(self):
        """speed can be negative or positive based on direction
        """

        logging.info(f"adding pedestrian local dynamics for scene {self.sceneId}")
        self._addLocalDynamicsForDf(self.getPedDataInSceneCoordinates())

        logging.info(f"adding other local dynamics for scene {self.sceneId}")
        self._addLocalDynamicsForDf(self.getOtherDataInSceneCoordinates())

        pass

    def _addLocalDynamicsForDf(self, df: pd.DataFrame):
        df["sceneXVelocity"] = TrajectoryUtils.getVelocitySeriesForAll(df, "sceneX", FPS)
        df["sceneYVelocity"] = TrajectoryUtils.getVelocitySeriesForAll(df, "sceneY", FPS)
        df["sceneXAcceleration"] = TrajectoryUtils.getAccelerationSeriesForAll(df, "sceneXVelocity", FPS)
        df["sceneYAcceleration"] = TrajectoryUtils.getAccelerationSeriesForAll(df, "sceneYVelocity", FPS)


    def _trimHeadAndTailForLocal(self):
        """ Must be called before building the scene track meta
        """
        logging.info(f"trimming pedestrian local data for scene {self.sceneId}")
        self._pedDataLocal = TrajectoryUtils.trimHeadAndTailForAll(self.getPedDataInSceneCoordinates())
        self._clippedPedData = self._pedDataLocal

        logging.info(f"trimming other local data for scene {self.sceneId}")
        self._otherDataLocal = TrajectoryUtils.trimHeadAndTailForAll(self.getOtherDataInSceneCoordinates())
        self._clippedOtherData = self._otherDataLocal


    def _buildSceneTrackMeta(self):

        pedDf = self.getPedDataInSceneCoordinates()
        otherDf = self.getOtherDataInSceneCoordinates()

        pedMeta = self.getMetaDictForDfs(pedDf)
        otherMeta = self.getMetaDictForDfs(otherDf)

        self._sceneTrackMeta = pd.concat(
            [pd.DataFrame(pedMeta), pd.DataFrame(otherMeta)], ignore_index=True)

    def getMetaDictForDfs(self, df: pd.DataFrame):


        meta = {
            "uniqueTrackId": [],
            "initialFrame": [],
            "finalFrame": [],
            "numFrames": [],
            "class": [],
            "horizontalDirection": [],
            "verticalDirection": []
        }

        if len(df) == 0:
            return meta

        ids = df["uniqueTrackId"].unique()

        for trackId in ids:
            trackDf = df[df["uniqueTrackId"] == trackId]
            # print(trackId, len(trackDf))
            firstRow = trackDf.iloc[0]
            lastRow = trackDf.iloc[1]

            vert, hort = TrajectoryUtils.getTrack_VH_Directions(
                trackDf, "sceneX", "sceneY")

            meta["uniqueTrackId"].append(trackId)
            meta["initialFrame"].append(firstRow["frame"])
            meta["finalFrame"].append(lastRow["frame"])
            meta["numFrames"].append(len(trackDf))
            if "class" in trackDf:
                meta["class"].append(firstRow["class"])
            else:
                meta["class"].append(TrackClass.Pedestrian.value)
            meta["horizontalDirection"].append(hort.value)
            meta["verticalDirection"].append(vert.value)

        return meta

    def getMeta(self):
        if self._sceneTrackMeta is None:
            self._buildSceneTrackMeta()
        return self._sceneTrackMeta

    def getTrackMeta(self, trackId) -> pd.Series:
        metaDf = self.getMeta()
        return metaDf[metaDf["uniqueTrackId"] == trackId].iloc[0]

    def get_VH_Directions(self, trackId) -> Tuple[TrackDirection, TrackDirection]:

        trackMeta = self.getTrackMeta(trackId)
        return (
            TrackDirection.createByValue(trackMeta["verticalDirection"]),
            TrackDirection.createByValue(trackMeta["horizontalDirection"])
        )

        pass

    def getPedDataInSceneCoordinates(self):
        if self._pedDataLocal is None:
            self._transformToLocalCoordinates()

        return self._pedDataLocal

    def getOtherDataInSceneCoordinates(self):
        if self._otherDataLocal is None:
            self._transformToLocalCoordinates()

        return self._otherDataLocal

    
    def getOtherByDirection(self, direction: TrackDirection) -> pd.DataFrame:
        otherDf = self.getOtherDataInSceneCoordinates()
        meta = self.getMeta()

        # pedIds in direction 
        classFilter = (meta["class"] != TrackDirection.Pedestrian.value)
        directionFilter = (meta["horizontalDirection"] == direction.value) | (meta["verticalDirection"] == direction.value)
        idsInDirection = meta[classFilter & directionFilter]

        return self.filterByIds(otherDf, idsInDirection)
        pass


    def getPedByDirection(self, direction: TrackDirection) -> pd.DataFrame:
        pedDf = self.getPedDataInSceneCoordinates()
        meta = self.getMeta()

        # pedIds in direction 
        classFilter = (meta["class"] == TrackDirection.Pedestrian.value)
        directionFilter = (meta["horizontalDirection"] == direction.value) | (meta["verticalDirection"] == direction.value)
        pedIdsInDirection = meta[classFilter & directionFilter]

        return self.filterByIds(pedDf, pedIdsInDirection)



    # region clipping

    def _clipPed(self, crossingOffset = CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS):
        logger.debug("clipping trajectories")
        scenePolygon = TrajectoryUtils.scenePolygon(
            self.sceneConfig, self.sceneConfig["boxWidth"], self.sceneConfig["roadWidth"] + crossingOffset)
        dfs = []
        for pedId in tqdm(self.uniquePedIds(), desc=f"clipping ped trajectories for scene # {self.sceneId} with width offset {crossingOffset}"):
            pedDf = self.getPedDfByUniqueTrackId(pedId)
            clippedDf = TrajectoryUtils.clipByRect(
                pedDf, "xCenter", "yCenter", "frame", scenePolygon)
            if TrajectoryUtils.length(clippedDf, "xCenter", "yCenter") < self.sceneConfig["roadWidth"] - 1:
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
            self.sceneConfig, self.sceneConfig["boxWidth"] + OTHER_CLIP_LENGTH, self.sceneConfig["roadWidth"] + 12) # needed for bikes
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
    
    #endregion

    def saveDataframes(self, pathPrefix: str):
        """
            saves clipped dataframes only
        """
        fpath = f"{pathPrefix}-scene-{self.sceneId}-pedestrians.csv"
        if os.path.exists(fpath):
            os.remove(fpath)

        self.getPedDataInSceneCoordinates().to_csv(fpath, index=False)

        fpath = f"{pathPrefix}-scene-{self.sceneId}-others.csv"
        if os.path.exists(fpath):
            os.remove(fpath)

        self.getOtherDataInSceneCoordinates().to_csv(fpath, index=False)

        fpath = f"{pathPrefix}-scene-{self.sceneId}-meta.csv"
        if os.path.exists(fpath):
            os.remove(fpath)

        self.getMeta().to_csv(fpath, index=False)
