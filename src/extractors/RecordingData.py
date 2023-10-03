import pandas as pd
from sortedcontainers import SortedList
from tools.TrajectoryUtils import TrajectoryUtils
from .SceneData import SceneData
from .TrackClass import TrackClass
from loguru import logger
import logging
from tqdm import tqdm
from .config import *

class RecordingData:

    def __init__(
        self,
        recordingId,
        recordingMeta,
        tracksMetaDf,
        tracksDf,
        backgroundImagePath=None,
        downSampleFps=FPS
    ):

        self.recordingId = recordingId
        self.recordingMeta = recordingMeta
        self.tracksMetaDf = tracksMetaDf
        self.tracksDf = tracksDf
        self.backgroundImagePath = backgroundImagePath
        self.fps = downSampleFps

        self._crossingDfByAnnotation = None
        self._crossingDfBySceneConfig = None
        self._otherDfBySceneConfig = None
        self._sceneData = {}

        self._trackIdClassMap = {}
        self._extractTrackIdClasses()
        self._downSampleByTrackLifeTime(toFPS=self.fps)

    @property
    def locationId(self):
        return self.recordingMeta["locationId"]

    @property
    def orthoPxToMeter(self):
        return self.recordingMeta["orthoPxToMeter"]

    def _downSampleByTrackLifeTime(self, fromFPS=ORIGINAL_FPS, toFPS=FPS):
        logging.info(f"Downsampling recording {self.recordingId} from {fromFPS} to {toFPS}")
        self.tracksDf = TrajectoryUtils.downSampleByTrackLifeTime(self.tracksDf, fromFPS, toFPS)

    def _extractTrackIdClasses(self):
        for _, row in self.tracksMetaDf.iterrows():
            self._trackIdClassMap[row["trackId"]] = row["class"]
        pass

    def getClass(self, trackId):
        return self._trackIdClassMap[trackId]

    def getDfByTrackIds(self, trackIds):
        criterion = self.tracksDf['trackId'].map(
            lambda trackId: trackId in trackIds)
        return self.tracksDf[criterion]

    def getIdsByClass(self, cls) -> SortedList:
        return SortedList(self.tracksMetaDf[self.tracksMetaDf['class'] == cls]['trackId'].tolist())

    def getPedIds(self) -> SortedList:
        return self.getIdsByClass(TrackClass.Pedestrian.value)

    def getCarIds(self) -> SortedList:
        return self.getIdsByClass(TrackClass.Car.value)

    def getBicycleIds(self) -> SortedList:
        return self.getIdsByClass(TrackClass.Bicycle.value)

    def getLargeVehicleIds(self) -> SortedList:
        return self.getIdsByClass(TrackClass.Truck_Bus.value)

    def getVehicleIds(self) -> SortedList:
        return self.getCarIds() + self.getLargeVehicleIds()

    def getDfByFrameSpan(self, start, end):
        return self.tracksDf[(self.tracksDf["frame"] >= start) & (self.tracksDf["frame"] <= end)]

    def getDfById(self, id):
        return self.tracksDf[self.tracksDf["trackId"] == id]

    def getDfByPedFrameSpan(self, pedId):
        """It returns all the rows that appears between the start and the end frame of the pedestrian. We probably should not use this as this does not reflect the crossing span.

        Args:
            pedId (_type_): _description_
        """
        # 1. get meta
        pedMeta = self.tracksMetaDf[self.tracksMetaDf['trackId'] == pedId]
        raise NotImplementedError("getDfByPedFrameSpan")

    def _getCrossingPedIdsByAnnotation(self) -> SortedList:
        if 'crossing' not in self.tracksMetaDf:
            raise Exception("crossing annotation not in tracksMetaDf")

        return SortedList(self.tracksMetaDf[(self.tracksMetaDf['class'] == 'pedestrian') & (self.tracksMetaDf['crossing'] == 'yes')]['trackId'].tolist())

    def getCrossingDfByAnnotations(self, sceneConfigs):
        if self._crossingDfByAnnotation is None:
            crossingIds = self._getCrossingPedIdsByAnnotation()
            self._crossingDfByAnnotation = self.getDfByTrackIds(crossingIds)

            # sceneIds = list(sceneConfigs.keys())
            # sceneDfs = []
            # # 1. Get all clipped scene crossing data in their local coordinate system
            # for sceneId  in sceneIds:
            #   logger.info(f"extracting crossing data for scene {sceneId} from recording {self.recordingId}")
            #   sC = sceneConfigs[str(sceneId)]
            #   df = self.getCrossingDfForScene(sceneId, sC, refresh=False, fps=FPS)
            #   if len(df) > 0:
            #     sceneDfs.append(df)

            # if len(sceneDfs) > 0:
            #   self._crossingDfBySceneConfig = pd.concat(sceneDfs, ignore_index=True)
            # else:
            #   logger.warning(f"No crossing data found for recording {self.recordingId}")
            #   self._crossingDfBySceneConfig = pd.DataFrame()

        raise Exception(
            "getCrossingDfByAnnotations does not have scene annotation")
        return self._crossingDfByAnnotation

    def getCrossingDfBySceneConfig(self, sceneConfigs, refresh=False, fps=FPS):
        logger.debug(
            f"getCrossingDfBySceneConfig from recording {self.recordingId}")

        if self._crossingDfBySceneConfig is None:
            sceneIds = list(sceneConfigs.keys())
            sceneDfs = []
            # 1. Get all clipped scene crossing data in their local coordinate system
            for sceneId in sceneIds:
                logger.info(
                    f"extracting crossing data for scene {sceneId} from recording {self.recordingId}")
                sC = sceneConfigs[str(sceneId)]
                df = self.getCrossingDfForScene(
                    sceneId, sC, refresh=False, fps=FPS)
                if len(df) > 0:
                    sceneDfs.append(df)

            if len(sceneDfs) > 0:
                self._crossingDfBySceneConfig = pd.concat(
                    sceneDfs, ignore_index=True)
            else:
                logger.warning(
                    f"No crossing data found for recording {self.recordingId}")
                self._crossingDfBySceneConfig = pd.DataFrame()

        return self._crossingDfBySceneConfig

    def getOtherDfBySceneConfig(self, sceneConfigs, refresh=False, fps=FPS):
        logger.debug(
            f"getOtherDfBySceneConfig from recording {self.recordingId}")

        if self._otherDfBySceneConfig is None:
            sceneIds = list(sceneConfigs.keys())
            sceneDfs = []
            # 1. Get all clipped scene crossing data in their local coordinate system
            for sceneId in sceneIds:
                logger.info(
                    f"extracting other data for scene {sceneId} from recording {self.recordingId}")
                sC = sceneConfigs[str(sceneId)]
                df = self.getOtherDfForScene(
                    sceneId, sC, refresh=False, fps=FPS)
                if len(df) > 0:
                    sceneDfs.append(df)

            if len(sceneDfs) > 0:
                self._otherDfBySceneConfig = pd.concat(
                    sceneDfs, ignore_index=True)
            else:
                logger.warning(
                    f"No crossing data found for recording {self.recordingId}")
                self._otherDfBySceneConfig = pd.DataFrame()

        return self._otherDfBySceneConfig

    def getSceneData(self, sceneId, sceneConfig, refresh=False, fps=FPS):
        """Do not use except for fast exploration. It's not used by the LocationData extractors. Always extracts by scene config

        Returns:
            _type_: _description_
        """

        if sceneId not in self._sceneData or refresh:

            pedData = self.getCrossingDfForScene(
                sceneId, sceneConfig, refresh=False, fps=FPS)
            if len(pedData) == 0:
                self._sceneData[sceneId] = None
            else:
                otherData = self.getOtherDfForScene(
                    sceneId, sceneConfig, refresh=False, fps=FPS)
                self._sceneData[sceneId] = SceneData(
                    self.locationId,
                    self.orthoPxToMeter,
                    sceneId,
                    sceneConfig,
                    sceneConfig["boxWidth"],
                    sceneConfig["roadWidth"],
                    pedData=pedData,
                    otherData=otherData,
                    backgroundImagePath=self.backgroundImagePath
                )

        return self._sceneData[sceneId]


    def getCrossingDfForScene(self, sceneId, sceneConfig, refresh=False, fps=FPS) -> pd.DataFrame:
        """Gets the pedestrian trajectories that crosses the scene

        Args:
            sceneId (_type_): _description_
            sceneConfig (_type_): _description_
            refresh (bool, optional): _description_. Defaults to False.
            fps (float, optional): _description_. Defaults to 2.5.

        Returns:
            pd.DataFrame: _description_
        """

        sceneDfs = []

        # create polygon
        scenePolygon = TrajectoryUtils.scenePolygon(
            sceneConfig, sceneConfig["boxWidth"], sceneConfig["roadWidth"] / 2)
        # create splines
        pedIds = self.getPedIds()
        for pedId in tqdm(pedIds, desc=f"recording-{self.recordingId}-scene{sceneId}-ped", leave=True, position=0):
            pedDf = self.getDfByTrackIds([pedId])
            df = TrajectoryUtils.getDfIfDfIntersect(
                sceneId=sceneId, sceneConfig=sceneConfig, scenePolygon=scenePolygon, df=pedDf)
            if df is not None:
                # df = TrajectoryUtils.downSample(df, ORIGINAL_FPS, fps)
                df["sceneId"] = sceneId
                sceneDfs.append(df)

        if len(sceneDfs) > 0:
            return pd.concat(sceneDfs, ignore_index=True)
        return pd.DataFrame()

    def getOtherDfForScene(self, sceneId, sceneConfig, refresh=False, fps=FPS) -> pd.DataFrame:
        otherDfs = []
        for otherClass in [TrackClass.Car, TrackClass.Bicycle, TrackClass.Truck_Bus]:
            otherDfs.append(
                self.getOtherDfForSceneByClass(
                    otherClass=otherClass,
                    sceneId=sceneId,
                    sceneConfig=sceneConfig,
                    refresh=refresh,
                    fps=fps
                )
            )
        return pd.concat(otherDfs, ignore_index=True)

    def getOtherDfForSceneByClass(self, otherClass: TrackClass, sceneId, sceneConfig, refresh=False, fps=FPS) -> pd.DataFrame:
        """Gets all but pedestrians that crosses the scene

        Args:
            sceneId (_type_): _description_
            sceneConfig (_type_): _description_
            refresh (bool, optional): _description_. Defaults to False.
            fps (float, optional): _description_. Defaults to 2.5.

        Returns:
            pd.DataFrame: _description_
        """
        otherDfs = []

        # create polygon
        scenePolygon = TrajectoryUtils.scenePolygon(
            sceneConfig, sceneConfig["boxWidth"], sceneConfig["roadWidth"] / 2)
        # create splines
        otherIds = self.getIdsByClass(otherClass.value)
        for otherId in tqdm(otherIds, desc=f"recording-{self.recordingId}-scene{sceneId}-{otherClass.value}-Ids", leave=True, position=0):
            inputDf = self.getDfByTrackIds([otherId])
            df = TrajectoryUtils.getDfIfDfIntersect(
                sceneId=sceneId, sceneConfig=sceneConfig, scenePolygon=scenePolygon, df=inputDf)
            if df is not None:
                # df = TrajectoryUtils.downSample(df, ORIGINAL_FPS, fps)
                df["sceneId"] = sceneId
                df["class"] = otherClass.value
                otherDfs.append(df)

        if len(otherDfs) > 0:
            return pd.concat(otherDfs, ignore_index=True)
        return pd.DataFrame()


    def getPedDf(self):
        pedIds = self.getPedIds()
        pedDfsInList = [self.getDfById(pedId) for pedId in pedIds]
        return pd.concat(pedDfsInList, ignore_index=True)