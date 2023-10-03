from extractors.loader import Loader
from extractors.LocationData import LocationData
from extractors.SceneData import SceneData
from extractors.TrackClass import TrackClass
from .TrajectoryUtils import TrajectoryUtils
from loguru import logger
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

class TrajectoryVisualizer:
    """for InD only"""

    def __init__(self, loader: Loader):

        self.loader = loader
        self.scale_down_factor = 12  # fixed for InD dataset only

    def initPlot(self, recordingId, title, backgroundImagePath=None):

        # Create figure and axes
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_size_inches(15, 8)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.00)
        # self.fig.canvas.set_window_title(title)
        self.ax.set_title(title)
        if backgroundImagePath is None:
            backgroundImagePath = self.loader.getBackgroundImagePath(
                recordingId)

        if backgroundImagePath and os.path.exists(backgroundImagePath):
            logger.info("Loading background image from {}",
                        backgroundImagePath)
            self.background_image = cv2.cvtColor(
                cv2.imread(backgroundImagePath), cv2.COLOR_BGR2RGB)
            (self.image_height,
             self.image_width) = self.background_image.shape[:2]
        else:
            logger.warning(
                "No background image given or path not valid. Using fallback black background.")
            self.image_height, self.image_width = 1700, 1700
            self.background_image = np.zeros(
                (self.image_height, self.image_width, 3), dtype="uint8")
        self.ax.imshow(self.background_image)

    def plot(self, orthoPxToMeter, tracksDf, style="m", xCol='xCenter', yCol='yCenter'):
        # TODO fixed to scale down factor of inD
        ortho_px_to_meter = orthoPxToMeter * self.scale_down_factor

        self.ax.plot(tracksDf[xCol] / ortho_px_to_meter, -
                     tracksDf[yCol] / ortho_px_to_meter, style)
        
        # plot direction
        lastRow = tracksDf.tail(1)
        endPoint = (lastRow[xCol] / ortho_px_to_meter,  - lastRow[yCol] / ortho_px_to_meter)
        self.ax.plot(endPoint[0], endPoint[1], marker='x', markerfacecolor=style[0])


    def showTrack(self, tracksDf, recordingMeta, trackId):
        self.initPlot(recordingMeta["recordingId"],
                      f"Trajectory for track {trackId}")

        pedTracksDf = tracksDf[tracksDf.trackId == trackId]
        self.plot(recordingMeta["orthoPxToMeter"], pedTracksDf)
        return pedTracksDf

    def showLocationCrossingTracks(self, locationData: LocationData):
        self.initPlot(
            locationData.recordingIds[0], f"Trajectories for location{locationData.locationId}")
        crossingDf = locationData.getCrossingDf()

        uniqueCrossingIds = locationData.getUniqueCrossingIds()
        for uniqueTrackId in uniqueCrossingIds:
            pedDf = locationData.getCrossingDfByUniqueTrackId(uniqueTrackId)
            self.plot(locationData.orthoPxToMeter, pedDf)

    def showLocationSceneData(self, sceneData: SceneData, onlyClipped=False, showLocal=False, showOthers=False, ids=None, offset=None, limit=100):

        self.initPlot(
            recordingId=list(sceneData.pedData.recordingId)[0],
            title=f"Trajectories for location{sceneData.locationId} and scene {sceneData.sceneId}",
            backgroundImagePath=sceneData.backgroundImagePath
        )

        # scene box
        
        self.plotSceneBox(sceneData)

        # show pedestrians

        # uniqueCrossingIds = sceneData.uniquePedIds()
        uniqueCrossingIds = ids
        if uniqueCrossingIds is None:
            uniqueCrossingIds = sceneData.uniquePedIds()
            
        if offset is not None:
            uniqueCrossingIds = uniqueCrossingIds[offset: offset+limit]
        for uniqueTrackId in uniqueCrossingIds:
            # if (ids is not None) and (uniqueTrackId not in ids):
            #     continue
            if not onlyClipped:
                pedDf = sceneData.getPedDfByUniqueTrackId(uniqueTrackId)
                if len(pedDf) == 0:
                    #this is a split one
                    pedDf = sceneData.getPedDfByUniqueTrackId(int(uniqueTrackId // 1000))
                self.plot(sceneData.orthoPxToMeter, pedDf)

            clippedDf = sceneData.getPedDfByUniqueTrackId(
                uniqueTrackId, clipped=True)
            self.plot(sceneData.orthoPxToMeter, clippedDf, style="c--")

            if showLocal:
                # localDf = sceneData.getPedDataInSceneCorrdinates()
                self.plot(sceneData.orthoPxToMeter, clippedDf,
                          xCol='sceneX', yCol='sceneY')

        # show others
        if showOthers:
            uniqueCrossingIds = sceneData.uniqueOtherIds()
            if offset is not None:
                uniqueCrossingIds = uniqueCrossingIds[offset: offset+limit]

            for uniqueTrackId in uniqueCrossingIds:
                if (ids is not None) and (uniqueTrackId not in ids):
                    continue
                if not onlyClipped:
                    otherDf = sceneData.getOtherDfByUniqueTrackId(
                        uniqueTrackId)
                    self.plot(sceneData.orthoPxToMeter, otherDf, style="w")

                clippedDf = sceneData.getOtherDfByUniqueTrackId(
                    uniqueTrackId, clipped=True)
                self.plot(sceneData.orthoPxToMeter, clippedDf, style="y")

                if showLocal:
                    # localDf = sceneData.getPedDataInSceneCorrdinates()
                    self.plot(sceneData.orthoPxToMeter, clippedDf,
                              style="w", xCol='sceneX', yCol='sceneY')
        plt.show()
    

    def showSceneProblems(self, sceneData: SceneData):
        self.initPlot(
            recordingId=list(sceneData.pedData.recordingId)[0],
            title=f"Trajectories for location{sceneData.locationId} and scene {sceneData.sceneId}",
            backgroundImagePath=sceneData.backgroundImagePath
        )
        self.plotSceneBox(sceneData)
        for problemClass in sceneData.problematicIds:
            problemIds = sceneData.problematicIds[problemClass]
            
            print(f"Showing {problemClass} problems")
            pedIds = sceneData.uniquePedIds()
            ohterIds = sceneData.uniqueOtherIds()
            # can be in both ped or other
            for problemId in problemIds:
                if problemId in pedIds:
                    # print("in pedIds?")
                    fullTrack = sceneData.getPedDfByUniqueTrackId(problemId)
                    clippedTrack = sceneData.getPedDfByUniqueTrackId(
                    problemId, clipped=True)
                else:
                    fullTrack = sceneData.getOtherDfByUniqueTrackId(problemId)
                    clippedTrack = sceneData.getOtherDfByUniqueTrackId(
                    problemId, clipped=True)
                
                # print(fullTrack, clippedTrack)
                if len(fullTrack) == 0:
                    raise ValueError(f"{problemClass} {problemId} has no trajectory!")
                self.plot(sceneData.orthoPxToMeter, fullTrack)
                self.plot(sceneData.orthoPxToMeter, clippedTrack)

            plt.show()



    
    def plotSceneBox(self, sceneData: SceneData):
        
        # plot scene bounding box before dynamics
        ortho_px_to_meter = sceneData.orthoPxToMeter * self.scale_down_factor
        outerPolygon = TrajectoryUtils.scenePolygon(sceneData.sceneConfig, sceneData.sceneConfig["boxWidth"], sceneData.sceneConfig["roadWidth"] + sceneData.CROSSING_CLIP_OFFSET_BEFORE_DYNAMICS)
        (X, Y) = outerPolygon.exterior.xy
        X = np.array(X)
        Y = np.array(Y)
        X /= ortho_px_to_meter
        Y /= -ortho_px_to_meter
        self.ax.plot(X, Y, "cornsilk")

        # plot scene bounding box
        ortho_px_to_meter = sceneData.orthoPxToMeter * self.scale_down_factor
        innerPolygon = TrajectoryUtils.scenePolygon(sceneData.sceneConfig, sceneData.sceneConfig["boxWidth"], sceneData.sceneConfig["roadWidth"] + sceneData.CROSSING_CLIP_OFFSET_AFTER_DYNAMICS)
        (X, Y) = innerPolygon.exterior.xy
        X = np.array(X)
        Y = np.array(Y)
        X /= ortho_px_to_meter
        Y /= -ortho_px_to_meter
        self.ax.plot(X, Y, "cornsilk")

        # plot scene center

        X = [sceneData.centerX / ortho_px_to_meter]
        Y = [sceneData.centerY / -ortho_px_to_meter]
        self.ax.plot(X, Y, marker='o', markersize=7,
                     markerfacecolor="yellow", markeredgecolor="black")

    def showLocalTrajectories(self, df, idCol, xCol, yCol):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        pedIds = df[idCol].unique()
        for pedId in pedIds:
            pedDf = df[df[idCol] == pedId]  
            plt.plot(pedDf[xCol], pedDf[yCol])
            # plot direction
            lastRow = pedDf.tail(1)
            endPoint = (lastRow[xCol] , lastRow[yCol])
            plt.plot(endPoint[0], endPoint[1], marker='x')

        ax.set_aspect('equal', adjustable='box')

    def showPedestrianTrajectoriesInRecording(self, recordingData):
        self.initPlot(
            recordingId=recordingData.recordingId,
            title=f"Pedestrian Trajectories for recording {recordingData.recordingId}"
            # backgroundImagePath=recordingData.backgroundImagePath
        )
        print(recordingData.backgroundImagePath)
        pedIds = recordingData.getPedIds()

        for pedId in pedIds:
            pedDf = recordingData.getDfById(pedId)
            self.plot(recordingData.orthoPxToMeter, pedDf)

        plt.show()