from .SceneData import SceneData
from .LocationData import LocationData
from tools.TrajectoryUtils import TrajectoryUtils
from .PedScenario import PedScenario, PedScenarioType
from .TrackClass import TrackClass
import pandas as pd
from shapely.geometry import LineString
from loguru import logger
from typing import List
from .config import *


class PedScenarioBuilder:

    """
        for each crossing uniqueTrackId
            for its recordingId, start and end 
                generate a scenario id,
                get all the rows overlapping the crossing frames
                filter out the trajectories that does not overlap with the scene config
                remove own frames
                extract types and ids of other tracks
                build PedScenario ob
    """

    def __init__(
            self
        ):

        self._nextId = 0
        self._scenarios = {} # grouped by scene id



    def getNextId(self):
        self._nextId += 1
        return self._nextId
    

    def buildFromLocationData(self, locationData: LocationData):
        sceneConfigs = locationData.getSceneConfig()
        
        sceneIds = list(sceneConfigs.keys())

        for sceneId in sceneIds:
            sceneConfig = sceneConfigs[sceneId]
            sceneData = locationData.getSceneData(
                sceneId, 
                sceneConfig["boxWidth"], 
                sceneConfig["roadWidth"]
            )
            self.buildFromSceneData(sceneData)


    def printSceneStats(self, sceneData: SceneData):
        # print all ped start end
        # print all other start end
        
        primaryPedIds = sceneData.uniqueClippedPedIds()
        for primaryPedId in primaryPedIds:
            primaryDf = sceneData.getClippedPedDfByUniqueTrackId(primaryPedId)
            recordingId, start, end, roadWidth = self.getRecordStartEndWidth(primaryDf)
            print("recordingId, primaryPedId,  start, end, roadWidth ", recordingId, primaryPedId, start, end, roadWidth )

    def buildFromSceneData(self, sceneData: SceneData):
        logger.info(f"building for scene {sceneData.sceneId}")

        primaryPedIds = sceneData.uniqueClippedPedIds()
        allPedDfs = sceneData.getClippedPedDfs()
        otherDf = sceneData.getClippedOtherDfs()

        self._scenarios[sceneData.sceneId] = []
        
        for primaryPedId in primaryPedIds:
            primaryDf = sceneData.getClippedPedDfByUniqueTrackId(primaryPedId).copy()
            recordingId, start, end, roadWidth = self.getRecordStartEndWidth(primaryDf)
            secondariesDf = self.getOtherScenarioTracksFromScene(
                otherSceneDf=otherDf,
                recordingId = recordingId,
                start=start,
                end=end
            )

            secondaryPedsDf = self.getOtherPedScenarioTracksFromScene(
                primaryDf=primaryDf,
                allPedDf=allPedDfs,
                recordingId = recordingId,
                start=start,
                end=end
            )

            secondariesDf = self.keepOtherIntersecting(primaryDf, secondariesDf)

            tags = self.getTagsFromSceneOtherDf(secondariesDf)

            primaryDf["class"] = "pedestrian"
            data = pd.concat([primaryDf, secondariesDf], ignore_index=True)

            self._scenarios[sceneData.sceneId].append(
                PedScenario(
                    sceneId=sceneData.sceneId,
                    scenarioId=self.getNextId(),
                    pedId=primaryPedId,
                    start=start,
                    end=end,
                    fps=FPS,
                    tags=tags,
                    data=data,
                    crossWalkLength=sceneData.boxHeight
                )
            )

   

    def getRecordStartEndWidth(self, trajDf: pd.DataFrame):
        """returns the start, end, and recordingId of a single trajectory

        Args:
            df (pd.DataFrame): Df must be for a single track, single record, and single scene

        Returns:
            _type_: _description_
        """
        head = trajDf.head(1).to_dict("records")[0]
        tail = trajDf.tail(1).to_dict("records")[0]
        return head["recordingId"], head["frame"], tail["frame"], head["roadWidth"]

    

    def getOtherScenarioTracksFromScene(self, otherSceneDf: pd.DataFrame, recordingId, start, end):
        """Scenario tracks. Returns a copy

        Args:
            otherSceneDf (pd.DataFrame): other trajectories from the same scene
            recordingId (_type_): _description_
            start (_type_): _description_
            end (_type_): _description_
            roadWidth (_type_): _description_
        """
        
        filter = (otherSceneDf["recordingId"] == recordingId) & (otherSceneDf["frame"] >= start) & (otherSceneDf["frame"] <= end)
        copied = otherSceneDf[filter].copy()
        copied.reset_index(drop=True)
        return copied

    
    # def isPedestrian(self, otherDf: pd.DataFrame):
    #     return self.getOtherType(otherDf) == TrackClass.Pedestrian.value

    # def isBicycle(self, otherDf: pd.DataFrame):
    #     return self.getOtherType(otherDf) == TrackClass.Bicycle.value

    # def isCar(self, otherDf: pd.DataFrame):
    #     return self.getOtherType(otherDf) == TrackClass.Car.value

    # def isLargeVehicle(self, otherDf: pd.DataFrame):
    #     return self.getOtherType(otherDf) == TrackClass.Truck_Bus.value

    
    def keepOtherIntersecting(self, primaryDf: pd.DataFrame, secondariesDf: pd.DataFrame):
        """_summary_

        Args:
            primaryDf (pd.DataFrame): primary pedestrian trajectory
            secondariesDf (pd.DataFrame): Other trajectories
        """

        uniqueClasses = secondariesDf["class"].unique()
        if "pedestrian" in uniqueClasses:
            raise Exception(f"Other dataframe has pedestrian class!")

        intersectingDfs = []
        otherTrackIds = secondariesDf["uniqueTrackId"].unique()
        for otherId in otherTrackIds:
            otherDf = secondariesDf[secondariesDf["uniqueTrackId"] == otherId]
            # if self.isPedestrian(otherDf):
            #     # keep even though they don't intersect with each other
            #     intersectingDfs.append(otherDf)

            if TrajectoryUtils.doPathsIntersect(primaryDf, otherDf):
                intersectingDfs.append(otherDf)

        
        return pd.concat(intersectingDfs, ignore_index=True)


    def getOtherPedScenarioTracksFromScene(self, 
                primaryDf: pd.DataFrame, 
                allPedDfs: pd.DataFrame,
                recordingId,
                start,
                end
            ):


        primaryPedId = primaryDf["uniqueTrackId"].iat[0]
        scenePedsDf = self.getOtherScenarioTracksFromScene(
            otherSceneDf=allPedDfs,
            recordingId = recordingId,
            start=start,
            end=end
        )

        # print(f"pedestrian with {recordingId, start, end}",  scenePedsDf["uniqueTrackId"].unique())
        primaryIndices = scenePedsDf[scenePedsDf["uniqueTrackId"] == primaryPedId].index
        scenePedsDf.drop(primaryIndices, inplace=True)
        scenePedsDf.reset_index(drop=True)
        return scenePedsDf
        


    def getTagsFromSceneOtherDf(self, primaryDf: pd.DataFrame, secondariesDf: pd.DataFrame):
        
        uniqueClasses = secondariesDf["class"].unique()
        if "pedestrian" in uniqueClasses:
            raise Exception(f"Other dataframe has pedestrian class!")

        otherTrackIds = secondariesDf["uniqueTrackId"].unique()
        tags = set([])

        # precompute splines
        primarySpline = TrajectoryUtils.dfToSplines(primaryDf)
        otherSplines = {}

        # if len(otherTrackIds) > 0:
        #     tags.append(PedScenarioType.)
        for otherId in otherTrackIds:
            otherDf = secondariesDf[secondariesDf["uniqueTrackId"] == otherId] # this is not right? need to filter by frame span
            otherSpline = TrajectoryUtils.dfToSplines(otherDf)
            otherSplines[otherId] = TrajectoryUtils.dfToSplines(otherSpline)

            # if TrackClass.isPedestrian(otherDf):
            #     tags += self.getInteractionTags(primarySpline, otherSpline)
            # else:
            tags.add(PedScenarioType.OnComingVehicle)
            if TrackClass.isCar(otherDf):
                tags.add(PedScenarioType.OnComingCar)
            elif TrackClass.isBicycle(otherDf):
                tags.add(PedScenarioType.OnComingBicyle)
            elif TrackClass.isLargeVehicle(otherDf):
                tags.add(PedScenarioType.OnComingLargeVehicle)
            else:
                raise Exception(f"Unknown class in other df {otherDf['class'].iat[0]}")
        
        return tags

    
    def getInteractionTags(self, primaryDf: pd.DataFrame, otherScenarioPedDf: pd.DataFrame):
        """_summary_

        Args:
            primaryDf (pd.DataFrame): _description_
            otherScenarioPedDf (pd.DataFrame): must only have pedestrians from the same scenario
        """
        
        if 'class' in otherScenarioPedDf:
            uniqueClasses = otherScenarioPedDf["class"].unique()
            if "pedestrian" not in uniqueClasses or len(uniqueClasses):
                raise Exception(f"Scenario ped df has non-pedestrian classes{' '.join(uniqueClasses)}!")

        otherTrackIds = otherScenarioPedDf["uniqueTrackId"].unique()
        tags = set([])

        # precompute splines
        primarySpline = TrajectoryUtils.dfToSplines(primaryDf)
        otherSplines = {}

        for otherId in otherTrackIds:
            otherDf = otherScenarioPedDf[otherScenarioPedDf["uniqueTrackId"] == otherId] # this is not right? need to filter by frame span
            otherSpline = TrajectoryUtils.dfToSplines(otherDf)
            otherSplines[otherId] = TrajectoryUtils.dfToSplines(otherSpline)
            tags += self.getInteractionTags(primarySpline, otherSpline)
        
        return tags

    

    def getInteractionTags(self, pedSpline1: LineString, pedSpline2: LineString) -> List[PedScenarioType]:
        # TODO interactions from https://github.com/vita-epfl/trajnetplusplustools/blob/master/trajnetplusplustools/interactions.py

        tags = set([])

        interacting = False
        if TrajectoryUtils.minSplineDistance(pedSpline1, pedSpline2) < 0.5: # 0.5 meter
            tags.add(PedScenarioType.Interacting) # TODO: not correct
            
            if TrajectoryUtils.sameDirectionSplines(pedSpline1, pedSpline2):
                tags.add(PedScenarioType.LeaderFollower) # TODO: Not correct
                tags.add(PedScenarioType.Group) # TODO: Not correct
            else:
                tags.add(PedScenarioType.CollisionAvoidance) # TODO: Not correct

        else:
            tags.add(PedScenarioType.NonInteracting) # TODO: not correct
        
        return tags






