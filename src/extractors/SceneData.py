import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from tools.TrajectoryUtils import TrajectoryUtils
from .TrackClass import TrackClass
from loguru import logger
from tqdm import tqdm
from .config import *
from .TrackDirection import TrackDirection
import os
from dill import dump, load
from typing import List, Tuple, Set
import logging
from collections import defaultdict

from tti_dataset_tools.TrajectoryTransformer import TrajectoryTransformer
from tti_dataset_tools.TrajectoryCleaner import TrajectoryCleaner
from tti_dataset_tools.ColMapper import ColMapper

# from tools.TrajectoryAnalyzer import TrajectoryAnalyzer


class SceneData:
    """SceneData only has crossing trajectories.

    """

    # @staticmethod
    # def create(
    #     self,
    #     locationId,
    #     orthoPxToMeter,
    #     sceneId,
    #     sceneConfig,
    #     boxWidth,
    #     boxHeight,
    #     pedData: pd.DataFrame,
    #     otherData: pd.DataFrame,
    #     backgroundImagePath=None
    # ):


    def __init__(
        self,
        locationId,
        orthoPxToMeter,
        sceneId,
        sceneConfig,
        boxWidth, # TODO unused in clipping. remove
        boxHeight, # TODO unused in clipping. remove
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

        # self.boxWidth = boxWidth
        # self.boxHeight = boxHeight
        self.polygon = TrajectoryUtils.scenePolygon(
            sceneConfig, boxWidth, boxHeight) # unused execpt visualizer

        self.pedData = pedData
        self._clippedPedData = None

        self.otherData = otherData
        self._clippedOtherData = None
        self._sceneTrackMeta = None

        # self.trajAnalyzer = TrajectoryAnalyzer(
        #     fps=FPS,
        #     idCol="uniqueTrackId",
        #     positionCols=("sceneX", "sceneY"),
        #     velocityCols=("xVelocity", "yVelocity"),
        #     accelerationCols=("xAcceleration", "yAcceleration")
        # )
        # print(len(self.pedData))

        self.CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS = CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS
        self.CROSSING_CLIP_OFFSET_AFTER_DYNAMICS = CROSSING_CLIP_OFFSET_AFTER_DYNAMICS

        self._isLocalTransformationDone = False
        self._isLocalInfomationBuilt = False

        self.warnings = []
        # self.problematicIds = {
        #     "ped": set([]),
        #     "fast-ped": set([]),
        #     "other": set([])
        # }
        self.problematicIds = defaultdict(lambda : set([]))
        
    
    def _deriveLocalCoordinateAndDynamics(self):
        self.reClip() # enables rebuilding
        # self.appendSceneIdToClipped()

        self._transformToLocalCoordinates()
        self._addLocalDynamics()
        
        self._trimHeadAndTailForLocal()
    
    
    def buildLocalInformation(self, transformer: TrajectoryTransformer, cleaner=TrajectoryCleaner, force=False):

        if force:
            self._isLocalTransformationDone = False
            self._isLocalInfomationBuilt = False

        # comment later
        # self._isLocalTransformationDone = False
        # self._isLocalInfomationBuilt = False

        if self._isLocalInfomationBuilt:
            return
        
            
        self.warnings = []
        self.problematicIds = defaultdict(lambda : set([]))

        self._dropWorldCoordinateColumns()

        # do cleaning on original trajectories
        self.cleanup(transformer, cleaner, force=force)

        self._deriveLocalCoordinateAndDynamics()


        # # redo clipping and deriving data again. # TODO it's too slow, do everything on original data
        # self._deriveLocalCoordinateAndDynamics()



        idsBefore = self.uniqueClippedPedIds()
        
        logging.info(f"Scene {self.sceneId}: clipping trimmed data")
        # print("here", self.CROSSING_CLIP_OFFSET_AFTER_DYNAMICS)
        self._clipPed(crossingOffset = self.CROSSING_CLIP_OFFSET_AFTER_DYNAMICS, onFull=False) # another pass as we had bigger offset to calculate dynamics
        idsAfter = self.uniqueClippedPedIds()

        idDiff = idsBefore - idsAfter
        if len(idDiff) > 0:
            self.warnings.append(f"Clipping after trimming lost {len(idDiff)} pedestrian tracks: {(idDiff)}")
            

        self._buildSceneTrackMeta()

        self._isLocalInfomationBuilt = True

        print("\n".join(self.warnings))
        print("problematic tracks:", self.problematicIds)
        pass

    def uniquePedIds(self) -> np.ndarray:
        return self.pedData.uniqueTrackId.unique()

    def uniqueClippedPedIds(self) -> Set[int]:
        clippedDf = self.getClippedPedDfs()
        if len(clippedDf) > 0:
            return set(clippedDf.uniqueTrackId.unique())
        return set([])

    def uniqueOtherIds(self) -> np.ndarray:
        return self.otherData.uniqueTrackId.unique()

    def uniqueClippedOtherIds(self) -> Set[int]:
        clippedDf = self.getClippedOtherDfs()
        if len(clippedDf) > 0:
            return set(clippedDf.uniqueTrackId.unique())
        return set([])

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
            ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1, errors="ignore")
        self.otherData = self.otherData.drop(
            ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1, errors="ignore")

    def _transformToLocalCoordinates(self):
        logging.info(f"Scene {self.sceneId}: transforming trajectories to scene coordinates")
        if self._isLocalTransformationDone:
            logging.info("Already transformed")
            return

        # translate and rotate.
        pedDf = self.getClippedPedDfs()
        self._transformDfToLocalCoordinates(pedDf)

        otherDf = self.getClippedOtherDfs()
        self._transformDfToLocalCoordinates(otherDf)

        self._isLocalTransformationDone = True
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
            # print("dhukbo")
            newPosition = TrajectoryUtils.transformPoint(
                translationMat, rotationMat, position)

            # row['sceneX'] = newPosition.x
            # row['sceneY'] = newPosition.y
            if newPosition.is_empty:
                print(f"Position : {position} NewPos : {newPosition}")
                print(f"Translation : {translationMat}")
                print(f"Rotation : {rotationMat}")
            sceneX.append(newPosition.x)
            sceneY.append(newPosition.y)

        df["sceneX"] = sceneX
        df["sceneY"] = sceneY
        return df

    def _addLocalDynamics(self):
        """speed can be negative or positive based on direction
        """

        logging.info(f"Scene {self.sceneId}: adding dynamics (velocity, acceleration) in scene coordinates")

        logging.debug(f"adding pedestrian local dynamics for scene {self.sceneId}")
        self._addLocalDynamicsForDf(self.getPedDataInSceneCoordinates())

        logging.debug(f"adding other local dynamics for scene {self.sceneId}")
        self._addLocalDynamicsForDf(self.getOtherDataInSceneCoordinates())

        pass

    def _addLocalDynamicsForDf(self, df: pd.DataFrame):
        # print(df.head())
        df["sceneXVelocity"] = TrajectoryUtils.getVelocitySeriesForAll(df, "sceneX", self.fps)
        df["sceneYVelocity"] = TrajectoryUtils.getVelocitySeriesForAll(df, "sceneY", self.fps)
        df["sceneXAcceleration"] = TrajectoryUtils.getAccelerationSeriesForAll(df, "sceneXVelocity", self.fps)
        df["sceneYAcceleration"] = TrajectoryUtils.getAccelerationSeriesForAll(df, "sceneYVelocity", self.fps)


    def _trimHeadAndTailForLocal(self):
        """ Must be called before building the scene track meta. It's required as the rolling velocity and acceleration do not have correct data for 4 frames.
        """
        logging.info(f"Scene {self.sceneId}: trimming head and tail")
        
        idsBefore = self.uniqueClippedPedIds()
        idsBeforeOther = self.uniqueClippedOtherIds()
        

        logging.debug(f"trimming pedestrian local data for scene {self.sceneId}")
        self._clippedPedData = TrajectoryUtils.trimHeadAndTailForAll(self.getPedDataInSceneCoordinates())

        logging.debug(f"trimming other local data for scene {self.sceneId}")
        self._clippedOtherData = TrajectoryUtils.trimHeadAndTailForAll(self.getOtherDataInSceneCoordinates())

        idsAfter = self.uniqueClippedPedIds()
        idsAfterOther = self.uniqueClippedOtherIds()

        idDiff = idsBefore - idsAfter
        if len(idDiff) > 0:
            self.warnings.append(f"Trimming after dynamics lost {len(idDiff)} pedestrian tracks: {(idDiff)}")

        idDiff = idsBeforeOther - idsAfterOther
        if len(idDiff) > 0:
            self.warnings.append(f"Trimming after dynamics lost {len(idDiff)} other tracks: {(idDiff)}")


    def _buildSceneTrackMeta(self):
        logging.info(f"Scene {self.sceneId}: building clipped track meta")

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
            lastRow = trackDf.iloc[-1]

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
        if not self._transformToLocalCoordinates:
            self._transformToLocalCoordinates()

        return self.getClippedPedDfs()

    def getOtherDataInSceneCoordinates(self):
        if not self._transformToLocalCoordinates:
            self._transformToLocalCoordinates()

        return self.getClippedOtherDfs()

    
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

    def assignNewTrackIdsToSplits(self, splitDfs: List[pd.DataFrame]) -> None:
        """Assumes all the splits have the same uniqueTrackId. Assumes max splits is 10

        Args:
            splitDfs (List[pd.DataFrame]): _description_

        Returns:
            pd.DataFrame: _description_
        """
        trackId = splitDfs[0].head(1)["uniqueTrackId"].iloc[0]
        if len(splitDfs) > 10:
            raise Exception(f"SceneData: track #{trackId} has been split ({len(splitDfs)}) times ({len(splitDfs)}). Cannot handle more than 10 tracks")
        trackId = trackId * 1000
        for splitDf in splitDfs:
            splitDf["uniqueTrackId"] = trackId
            trackId += 1
        
        pass    


    def hasValidCrossingLength(self, clippedDf:pd.DataFrame, trackId:int, trackClass:str, minLength:float, maxLength: float):

        trackLength = TrajectoryUtils.length(clippedDf, "xCenter", "yCenter")
        if (len(clippedDf) < 3) or (trackLength < minLength):
            logging.debug(
                f"{trackClass} {trackId}: Disregarding as the length {trackLength} is too short or rows too less ({len(clippedDf)})")
            self.warnings.append(
                f"{trackClass} {trackId}: Disregarding as the length {trackLength} is too short or rows too less ({len(clippedDf)})")
            return False

        elif trackLength > (maxLength):
            logging.warn(
                f"{trackClass} {trackId}: Disregarding as the length {trackLength} is too long)")
            self.warnings.append(
                f"{trackClass} {trackId}: Disregarding as the length {trackLength} is too long")
            return False
        
        return True


    def _clipTrack(self, trackDf: pd.DataFrame, scenePolygon:box, trackClass:str, minLength: float, maxLength: float) -> List[pd.DataFrame]:

            if len(trackDf) == 0: # means not in clipped df
                return []

            trackId = trackDf.head(1)["uniqueTrackId"].iloc[0]

            validClips = []

            if len(trackDf) < 3: 
                logging.debug(f"{trackClass} {trackId}: trajectory is too short ({len(trackDf)}) to be clipped")
                self.warnings.append(f"{trackClass} {trackId}: trajectory is too short ({len(trackDf)}) to be clipped")
            else:
                clippedDfs = TrajectoryUtils.clipByRectWithSplits(
                                trackDf, 
                                xCol="xCenter",
                                yCol="yCenter",
                                frameCol="frame",
                                rect=scenePolygon
                            )
                # clippedDf, exitCount = TrajectoryUtils.clipByRect(
                #     trackDf, "xCenter", "yCenter", "frame", scenePolygon)

                if len(clippedDfs) == 0:
                    logging.debug(f"{trackClass} {trackId}: ERROR: No clipped trajectory")
                    # raise Exception(f"No clipped trajectory for {trackClass} {trackId}")
                    self.warnings.append(f"{trackClass} {trackId}: ERROR: No clipped trajectory")
                
                else:
                
                    if len(clippedDfs) > 1:
                        self.warnings.append(f"{trackClass} {trackId}: enters the scene {len(clippedDfs)} times")

                    for clippedDf in clippedDfs:
                        if self.hasValidCrossingLength(clippedDf, trackId, trackClass, minLength, maxLength):
                            # clippedDf.reset_index(drop=True)
                            validClips.append(clippedDf)



                        
            if len(validClips) == 0:
                self.warnings.append(f"{trackClass} {trackId}: ERROR: No valid clipped trajectory")
                self.problematicIds[trackClass].add(trackId)

            return validClips


    def _clipPed(self, crossingOffset, onFull = True, ids=None):
        
        logger.debug("clipping trajectories")

        scenePolygon = TrajectoryUtils.scenePolygon(
            self.sceneConfig, self.sceneConfig["boxWidth"] + BOX_WIDTH_OFFSET, self.sceneConfig["roadWidth"] + crossingOffset)

        # logging.info(f"clipping trajectories with scene polygon {scenePolygon}")
        if ids is None:
            ids = self.uniquePedIds()
        dfs = []
        for pedId in tqdm(ids, desc=f"clipping ped trajectories for scene # {self.sceneId} with width offset {crossingOffset}"):
            pedDf = self.getPedDfByUniqueTrackId(pedId, clipped = not onFull)


            clippedDfs = self._clipTrack(
                trackDf=pedDf,
                scenePolygon=scenePolygon,
                trackClass=TrackClass.Pedestrian.value,
                # minLength=self.sceneConfig["roadWidth"] * 0.8,
                minLength=0.1,
                maxLength=(self.sceneConfig["roadWidth"] + crossingOffset) * 3,
            )


            if len(clippedDfs) > 1:
                self.assignNewTrackIdsToSplits(clippedDfs)

            # print(pedDf)
            # print(len(clippedDfs))
            # print(len(clippedDfs[0]), len(clippedDfs[1]))
            # print(clippedDfs[0]["uniqueTrackId"].unique(), clippedDfs[1]["uniqueTrackId"].unique())

            # print("before", len(dfs))
            if len(clippedDfs) > 0:
                dfs.extend(clippedDfs)
            else:
                self.warnings.append(
                    f"Ped {pedId}: is lost due to clipping. Check raw data")
                self.problematicIds[TrackClass.Pedestrian].add(pedId)

            # print("after", len(dfs))

            
        if len(dfs) == 0:
            """No pedData"""
            self._clippedPedData = pd.DataFrame()
        else:
            self._clippedPedData = pd.concat(dfs, ignore_index=True)

    def _clipOther(self, onFull = True):
        logger.debug("clipping other trajectories")
        # we will clip with the bounding box + 50 meters
        scenePolygon = TrajectoryUtils.scenePolygon(
            self.sceneConfig, self.sceneConfig["boxWidth"] + OTHER_CLIP_LENGTH, self.sceneConfig["roadWidth"] + 12) # needed for bikes
        dfs = []
        for otherId in tqdm(self.uniqueOtherIds(), desc=f"clipping other trajectories for scene # {self.sceneId}"):

            otherDf = self.getOtherDfByUniqueTrackId(otherId, clipped = not onFull)
            
            clippedDfs = self._clipTrack(
                trackDf=otherDf,
                scenePolygon=scenePolygon,
                trackClass=TrackClass.getTrackType(otherDf),
                # minLength=self.sceneConfig["roadWidth"], 
                minLength=0.2, 
                maxLength=self.sceneConfig["boxWidth"] + OTHER_CLIP_LENGTH * 3, 
            )

            if len(clippedDfs) > 1:
                self.assignNewTrackIdsToSplits(clippedDfs)

            if len(clippedDfs) > 0:
                dfs.extend(clippedDfs)
            else:
                self.warnings.append(
                    f"Other {otherId}: is lost due to clipping. Check raw data")
                self.problematicIds[TrackClass.getTrackType(otherDf)].add(otherId)



        if len(dfs) == 0:
            """No Data"""
            self._clippedOtherData = pd.DataFrame()
        else:
            self._clippedOtherData = pd.concat(dfs, ignore_index=True)

    def reClip(self):
        logging.info(f"Scene {self.sceneId}: clipping original data")
        self._clipPed(crossingOffset = self.CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS)
        self._clipOther()

    def appendSceneIdToClipped(self):
        clippedDf = self.getClippedPedDfs()
        clippedDf["uniqueTrackId"] = int(self.sceneId) * 10000000 + clippedDf["uniqueTrackId"] 

        clippedDf = self.getClippedOtherDfs()
        clippedDf["uniqueTrackId"] = int(self.sceneId) * 10000000 + clippedDf["uniqueTrackId"] 


    def getClippedPedDfs(self):
        if self._clippedPedData is None:
            # TO-DO: will replace CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS with self.CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS
            self._clipPed(crossingOffset = self.CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS)

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

        exportedAttrs = ["recordingId", "frame", "uniqueTrackId", "sceneId", "roadWidth", "sceneX", "sceneY", "sceneXVelocity", "sceneYVelocity", "sceneXAcceleration", "sceneYAcceleration"]
        self.getPedDataInSceneCoordinates()[exportedAttrs].to_csv(fpath, index=False)

        fpath = f"{pathPrefix}-scene-{self.sceneId}-others.csv"
        if os.path.exists(fpath):
            os.remove(fpath)

        exportedAttrs = ["recordingId", "frame", "uniqueTrackId", "class", "width", "length", "sceneId", "roadWidth", "sceneX", "sceneY", "sceneXVelocity", "sceneYVelocity", "sceneXAcceleration", "sceneYAcceleration"]
        self.getOtherDataInSceneCoordinates()[exportedAttrs].to_csv(fpath, index=False)

        fpath = f"{pathPrefix}-scene-{self.sceneId}-meta.csv"
        if os.path.exists(fpath):
            os.remove(fpath)

        self.getMeta().to_csv(fpath, index=False)

    #region clean up

    def cleanup(self, transformer: TrajectoryTransformer, cleaner: TrajectoryCleaner, force=False):
        self.moveOutlierPedsToOthers(transformer, cleaner, force=force)

    def moveOutlierPedsToOthers(self, transformer: TrajectoryTransformer, cleaner: TrajectoryCleaner, force=False):
        """
        fast_pedestrian
        """
        if self._isLocalInfomationBuilt and not force:
            raise Exception(f"Local information already build, cannot move outliers")

        outlierClass = 'fast_pedestrian'
        logging.info(f"SceneData {self.sceneId}: moving outlier peds to others. We should only find outliers in the clipped trajectories?")

        # pedDf = self.getPedDataInSceneCoordinates()
        # otherDf = self.getOtherDataInSceneCoordinates()

        # derive speed
        if "speed" not in self.pedData:
            transformer.deriveSpeed(self.pedData) # we need to call them again
            # transformer.deriveSpeed(otherDf) # we need to call them again
        # get outliers
        # move outliers to others
        # TODO update meta

        outlierPedIds = cleaner.getOutliersBySpeed(self.pedData, byIQR=False, returnVals=False)
        outlierDfs = []
        self.problematicIds[outlierClass] = set([])
        for outlierPedId in outlierPedIds:
            self.problematicIds[outlierClass].add(outlierPedId)
            outlierDf = self.pedData[self.pedData["uniqueTrackId"] == outlierPedId].copy() # original data
            outlierDf["class"] = outlierClass
            outlierDfs.append(outlierDf)
            self.warnings.append(
                f"{outlierClass} {outlierPedId}: moving {outlierPedId} to others as speed is unrealistic {outlierDf['speed']}")
        
        self.otherData = pd.concat([self.otherData] + outlierDfs, ignore_index=True)
        
        # drop from ped
        criterion = self.pedData["uniqueTrackId"].map(lambda trackId: trackId not in outlierPedIds)
        self.pedData = self.pedData[criterion].copy()

        pass

    #endregion