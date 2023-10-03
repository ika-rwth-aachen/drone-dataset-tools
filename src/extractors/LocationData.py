from typing import List
import pandas as pd
from sortedcontainers import SortedList
from loguru import logger
from tools.UnitUtils import UnitUtils
from tools.TrajectoryUtils import TrajectoryUtils
from .SceneData import SceneData
from tqdm import tqdm
from dill import dump, load
from datetime import datetime
import os
import functools
from .config import *
import logging


from tti_dataset_tools.TrajectoryTransformer import TrajectoryTransformer
from tti_dataset_tools.TrajectoryCleaner import TrajectoryCleaner
from tti_dataset_tools.ColMapper import ColMapper


class LocationData:

    def __init__(self, 
        locationId, 
        recordingIds, 
        recordingDataList, 
        fps: int,
        useSceneConfigToExtract=True, 
        precomputeSceneData=True,
        backgroundImagePath = None,
    ):
        """_summary_

        Args:
            locationId (_type_): _description_
            recordingIds (_type_): _description_
            recordingDataList (_type_): _description_
            useSceneConfigToExtract (bool, optional): We extract data in two ways:. Defaults to False.
            precomputeSceneData (bool, optional): extracts data. Defaults to True.
        """

        self.locationId = locationId
        self.recordingIds = recordingIds
        self.recordingDataList = recordingDataList

        self.useSceneConfigToExtract = useSceneConfigToExtract
        self.backgroundImagePath = backgroundImagePath

        self.recordingMetaList = [
            recordingData.recordingMeta for recordingData in recordingDataList]
        self.validateRecordingMeta()

        self.frameRate = self.recordingMetaList[0]["frameRate"]
        self.fps = fps
        self.orthoPxToMeter = self.recordingMetaList[0]["orthoPxToMeter"]

        # Trajectory transformer
        self.createTransformerCleaner()

        # cache
        self._crossingDf = None
        self._crossingIds = None

        self._otherDf = None
        self._otherIds = None

        self._sceneData = {}

        self._mergedSceneDfs = {}

        if precomputeSceneData:
            self._precomputeSceneData()

    def createTransformerCleaner(self):
        colMapper = ColMapper(
                idCol='uniqueTrackId', 
                # xCol='sceneX', 
                # yCol='sceneY',
                # xVelCol='xVelocity', 
                # yVelCol='xVelocity', 
                xCol='xCenter', 
                yCol='yCenter',
                xVelCol='xVelocity', 
                yVelCol='xVelocity', 
                speedCol='speed',
                fps=25
            )
        self.transformer = TrajectoryTransformer(colMapper)
        self.cleaner = TrajectoryCleaner(
            colMapper = colMapper,
            minSpeed = 0.0,
            maxSpeed = PED_MAX_SPEED,
            minYDisplacement = 5.0,
            maxXDisplacement = 8.0,
        )


    def validateRecordingMeta(self):
        sameValueFields = [
            "locationId",
            "frameRate",
            # "latLocation",
            # "lonLocation",
            # "xUtmOrigin",
            # "yUtmOrigin",
            "orthoPxToMeter"
        ]

        firstMeta = self.recordingMetaList[0]
        for otherMeta in self.recordingMetaList:
            for field in sameValueFields:
                if firstMeta[field] != otherMeta[field]:
                    raise Exception(
                        f"{field} value mismatch for {firstMeta['recordingId']} and {otherMeta['recordingId']}")

    def summary(self):

        summary = {
            "#original frameRate": self.frameRate,
            "#crossing trajectories": len(self.getUniqueCrossingIds()),
            "#scene trajectories": functools.reduce(lambda acc, new: acc + new, [SceneData.clippedPedSize() for SceneData in self._sceneData.values() if SceneData is not None])
        }

        for sceneId in self._sceneData.keys():
            if self._sceneData[sceneId] is not None:
                summary[f"scene#{sceneId} peds"] = self._sceneData[sceneId].clippedPedSize()
                summary[f"scene#{sceneId} others"] = self._sceneData[sceneId].clippedOtherSize()

        return summary

    # region extraction
    def getUniqueCrossingIds(self):
        """
        returns unique pedestrian ids
        """

        if self._crossingIds is None:
            self._crossingIds = SortedList()
            crossingDf = self.getCrossingDf()
            self._crossingIds.update(crossingDf.uniqueTrackId.unique())

        return self._crossingIds

    def getUniqueOtherIds(self):
        """
        returns unique pedestrian ids
        """

        if self._otherIds is None:
            self._otherIds = SortedList()
            crossingDf = self.getOtherDf()
            self._otherIds.update(crossingDf.uniqueTrackId.unique())

        return self._otherIds

    def getCrossingDfByUniqueTrackId(self, uniqueTrackId):
        return self.getCrossingDfByUniqueTrackIds([uniqueTrackId])

    def getCrossingDfByUniqueTrackIds(self, uniqueTrackIds):
        crossingDf = self.getCrossingDf()
        criterion = crossingDf['uniqueTrackId'].map(
            lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
        return crossingDf[criterion]

    def getOtherDfByUniqueTrackId(self, uniqueTrackId):
        return self.getOtherDfByUniqueTrackIds([uniqueTrackId])

    def getOtherDfByUniqueTrackIds(self, uniqueTrackIds):
        otherDf = self.getOtherDf()
        criterion = otherDf['uniqueTrackId'].map(
            lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
        return otherDf[criterion]


    def getRecordingCrossingDf(self, recordingData):
        if self.useSceneConfigToExtract:
            crossingDf = recordingData.getCrossingDfBySceneConfig(
                self.getSceneConfig())
        else:
            crossingDf = recordingData.getCrossingDfByAnnotations()  # it does not have scene id!
        return crossingDf

    def getRecordingOtherDf(self, recordingData):
        return recordingData.getOtherDfBySceneConfig(self.getSceneConfig())

    def getCrossingDf(self):
        """returns crossing data for all the scenes and recordings

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        if self._crossingDf is None:
            crossingDfs = []
            for recordingData in tqdm(self.recordingDataList, desc="crossing recording", position=0):
                try:
                    crossingDf = self.getRecordingCrossingDf(recordingData)
                    logger.info(
                        f"got crossing df for {recordingData.recordingId}")

                    if "uniqueTrackId" not in crossingDf:
                        raise Exception(
                            f"{recordingData.recordingId} does not have uniqueTrackId")

                    if len(crossingDf) == 0:
                        logging.warn("Recording {recordingData.recordingId} has no crossing data!")
                    else:
                        crossingDfs.append(crossingDf)

                except Exception as e:
                    logger.warning(
                        f"{recordingData.recordingId} has exception: {e}")
                    # raise e

                # logger.warning("Breaking after processing the first recording")
                # break

            self._crossingDf = pd.concat(crossingDfs, ignore_index=True)

        return self._crossingDf

    def getOtherDf(self):
        """returns other data for all the scenes and recordings

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        if self._otherDf is None:
            otherDfs = []
            for recordingData in tqdm(self.recordingDataList, desc="other recording", position=0):
                try:
                    otherDf = self.getRecordingOtherDf(recordingData)
                    logger.info(
                        f"got other df for {recordingData.recordingId}")
                    if "uniqueTrackId" not in otherDf:
                        raise Exception(
                            f"{recordingData.recordingId} does not have uniqueTrackId")
                    otherDfs.append(otherDf)
                except Exception as e:
                    logger.warning(
                        f"{recordingData.recordingId} has exception: {e}")
                    # raise e

                # logger.warning("Breaking after processing the first recording")
                # break

            self._otherDf = pd.concat(otherDfs, ignore_index=True)

        return self._otherDf
    # endregion

    # region scene
    def _precomputeSceneData(self):
        sceneConfigs = self.getSceneConfig()
        sceneIds = list(sceneConfigs.keys())

        for sceneId in sceneIds:
            logger.info(f"Precomputing sceneData for {sceneId}")
            sceneConfig = sceneConfigs[sceneId]
            self.getSceneData(
                sceneId, 
                sceneConfig["boxWidth"], 
                sceneConfig["roadWidth"]
            )
    
    def buildLocalInformationForScenes(self):
        self.createTransformerCleaner()
        sceneConfigs = self.getSceneConfig()
        sceneIds = list(sceneConfigs.keys())

        for sceneId in sceneIds:
            logger.info(f"Precomputing sceneData for {sceneId}")
            sceneConfig = sceneConfigs[sceneId]
            sceneData = self.getSceneData(
                sceneId, 
                sceneConfig["boxWidth"], 
                sceneConfig["roadWidth"],
                # refresh=True
            ) # just to be save if precomputation was not done
            if sceneData is not None:
                sceneData.buildLocalInformation(self.transformer, self.cleaner, force=True)
                
            



    def getSceneConfig(self):
        return UnitUtils.getLocationSceneConfigs("ind", self.locationId)

    def getSceneCrossingDf(self, sceneId, boxWidth, boxHeight) -> pd.DataFrame:

        sceneId = str(sceneId)

        if self.useSceneConfigToExtract:
            crossingDf = self.getCrossingDf()
            return crossingDf[crossingDf["sceneId"] == sceneId].copy().reset_index()

        logger.info(
            f"collecting scene crossing data from annotated data for scene {sceneId}")

        sceneDfs = []
        sceneConfig = self.getSceneConfig()[sceneId]

        # create polygon
        scenePolygon = TrajectoryUtils.scenePolygon(
            sceneConfig, boxWidth, boxHeight)
        # create splines
        crossingIds = self.getUniqueCrossingIds()
        for crossingId in crossingIds:
            pedDf = self.getCrossingDfByUniqueTrackId(crossingId)
            trajSpline = TrajectoryUtils.dfToSplines(
                pedDf, "xCenter", "yCenter", 1)
            if TrajectoryUtils.doesIntersect(scenePolygon, trajSpline):
                pedDf = pedDf.copy()  # we can modify without concern now
                pedDf["sceneId"] = sceneId
                sceneDfs.append(pedDf)

        return pd.concat(sceneDfs, ignore_index=True)

    def getSceneOtherDf(self, sceneId) -> pd.DataFrame:

        sceneId = str(sceneId)
        otherDf = self.getOtherDf()
        return otherDf[otherDf["sceneId"] == sceneId].copy().reset_index()
    
    def getSceneIds(self) -> List[str]:
        return list(self._sceneData.keys())

    def getSceneData(self, sceneId, boxWidth=6, boxHeight=6, refresh=False, fps=FPS) -> SceneData:
        """_summary_

        Args:
            sceneId (_type_): scene id from scene config file
            boxWidth (int, optional): width of the bounding box to filter unrelated trajectories. Runs along the road's length. Defaults to 6.
            boxHeight (int, optional): height of the bounding box to filter unrelated trajectories. Runs along the road's width. Defaults to 6.
            refresh (bool, optional): force re-filter when bounding box changes. Results are cached when run. Defaults to False.
            fps (float, optional): frame rate conversion from 25. Defaults to 2.5.

        Returns:
            SceneData: _description_
        """

        sceneId = str(sceneId)
        if sceneId not in self._sceneData or refresh:

            otherData = self.getSceneOtherDf(sceneId)
            pedData = self.getSceneCrossingDf(sceneId, boxWidth, boxHeight)
            
            if len(pedData) == 0:
                self._sceneData[sceneId] = None
                logging.warning(f"location {self.locationId} do not have any pedestrian data for scene {sceneId}!")
            else:
                sceneConfig = self.getSceneConfig()[str(sceneId)]
                self._sceneData[sceneId] = SceneData(
                    self.locationId,
                    self.orthoPxToMeter,
                    sceneId,
                    sceneConfig,
                    boxWidth,
                    boxHeight,
                    pedData=pedData,
                    otherData=otherData,
                    backgroundImagePath=self.backgroundImagePath
                )

        return self._sceneData[sceneId]

    def mergeScenesByRoadWidth(self, refresh=False):
        """
          merges local coordinates
        """

        if len(self._mergedSceneDfs) > 0 and not refresh:
            return self._mergedSceneDfs

        sceneConfigs = self.getSceneConfig()
        sceneIds = list(sceneConfigs.keys())

        groups = {}
        for sceneId in sceneIds:
            sceneConfig = sceneConfigs[sceneId]
            if sceneConfig["roadWidth"] not in groups:
                groups[sceneConfig["roadWidth"]] = []
            groups[sceneConfig["roadWidth"]].append(
                self.getSceneData(sceneId, 0, 0))

        for roadWidth in tqdm(groups, desc="merging scenes"):
            groupDfs = []
            group = groups[roadWidth]
            for SceneData in group:
                sceneLocalDf = SceneData.getPedDataInSceneCorrdinates()
                groupDfs.append(sceneLocalDf[[
                                "frame", "uniqueTrackId", "sceneX", "sceneY", "sceneId", "recordingId"]].copy())
            groupDf = pd.concat(groupDfs, ignore_index=True)
            # groupDf["roadWidth"] = roadWidth # already done when extracting by scene config, or bulding scene data
            self._mergedSceneDfs[roadWidth] = groupDf

        return self._mergedSceneDfs

    # endregion

    # region third party formats

    def getCrossingDataForTransformerNoMerge(self):
        """
        csv with frame, ped, x, y
        """
        sceneConfigs = self.getSceneConfig()
        sceneIds = list(sceneConfigs.keys())

        sceneDfs = []

        # 1. Get all clipped scene crossing data in their local coordinate system
        for sceneId in sceneIds:
            sceneConfig = sceneConfigs[str(sceneId)]
            SceneData = self.getSceneData(
                sceneId, sceneConfig["boxWidth"], sceneConfig["roadWidth"])
            sceneLocalDf = SceneData.getPedDataInSceneCorrdinates()
            if len(sceneLocalDf) > 0:
                sceneDfs.append(sceneLocalDf[[
                                "frame", "uniqueTrackId", "sceneX", "sceneY", "sceneId", "recordingId"]].copy())

        allSceneDf = pd.concat(sceneDfs, ignore_index=True)
        # 2. create unique integer ped ids (they are already int)
        # self._createUniqueIntegerPedId(allSceneDf)
        # 3. Augment data (flipping)
        # 4. fps conversion?

        # last: rename
        allSceneDf.rename(columns={
            "uniqueTrackId": "ped",
            "sceneX": "x",
            "sceneY": "y"
        })
        return allSceneDf

    def getCrossingDataForTransformer(self, refresh=False):
        allSceneDfs = {}
        mergedSceneDfs = self.mergeScenesByRoadWidth(refresh=refresh)
        for roadWidth in mergedSceneDfs:
            allSceneDfs[roadWidth] = mergedSceneDfs[roadWidth].copy().rename(columns={
                "uniqueTrackId": "ped",
                "sceneX": "x",
                "sceneY": "y"
            })
        return allSceneDfs

    # endregion

    # region cache

    def madeLocationDir(self, outputDir):
        locDir = os.path.join(outputDir, f"location-{self.locationId}")
        os.makedirs(locDir, exist_ok=True)
        return locDir

    def saveCrossingDf(self, outputDir):

        locDir = self.madeLocationDir(outputDir)
        date_time = datetime.now().strftime("%Y-%m-%d")

        fpath = os.path.join(locDir, f"{date_time}-fps-{self.fps}-crossing.csv")
        if os.path.exists(fpath):
            os.remove(fpath)
        crossingDf = self.getCrossingDf()
        crossingDf.to_csv(fpath, index=False)

        fpath = os.path.join(locDir, f"{date_time}-fps-{self.fps}-other.csv")
        if os.path.exists(fpath):
            os.remove(fpath)
        otherDf = self.getOtherDf()
        otherDf.to_csv(fpath, index=False)

        pass

    def save(self, outputDir):

        locDir = self.madeLocationDir(outputDir)
        date_time = datetime.now().strftime("%Y-%m-%d")

        fpath = os.path.join(locDir, f"{date_time}-fps-{self.fps}-all.dill")
        if os.path.exists(fpath):
            os.remove(fpath)
        with open(fpath, "wb") as fp:
            dump(self, fp)
            logger.info(f"saved to {fpath}")

    def saveSceneDataOnly(self, outputDir):

        locDir = self.madeLocationDir(outputDir)
        date_time = datetime.now().strftime("%Y-%m-%d")

        for sceneId in self._sceneData:
            if self.getSceneData(sceneId) is None:
                logging.warn(f"No scene data for {sceneId}")
                continue


            # dataframes
            dfPrefix = f"{date_time}-fps-{self.fps}"
            pathPrefix = os.path.join(locDir, dfPrefix)
            self.getSceneData(sceneId).saveDataframes(pathPrefix)

            # whole thing as dill
            fname = f"{date_time}-fps-{self.fps}-scene-{sceneId}.dill"
            fpath = os.path.join(locDir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
            with open(fpath, "wb") as fp:
                dump(self.getSceneData(sceneId), fp)
                logger.info(f"saved to {fpath}")



    @staticmethod
    def load(locDir, fname=None):

        if fname is None:
            fname = "all.dill"

        fpath = os.path.join(locDir, fname)
        logger.info(f"reading from {fpath}")
        with open(fpath, "rb") as fp:
            return load(fp)
    # endregion
