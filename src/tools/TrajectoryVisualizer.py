from extractors.loader import Loader
from extractors.LocationData import LocationData
from extractors.SceneData import SceneData
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

    def plot(self, orthoPxToMeter, tracksDf, style="r", xCol='xCenter', yCol='yCenter'):
        # TODO fixed to scale down factor of inD
        ortho_px_to_meter = orthoPxToMeter * self.scale_down_factor

        self.ax.plot(tracksDf[xCol] / ortho_px_to_meter, -
                     tracksDf[yCol] / ortho_px_to_meter, style)
        
        # plot direction
        lastRow = tracksDf.tail(1)
        endPoint = (lastRow[xCol] / ortho_px_to_meter,  - lastRow[yCol] / ortho_px_to_meter)
        self.ax.plot(endPoint[0], endPoint[1], marker='x')


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

    def showLocationSceneData(self, sceneData: SceneData, onlyClipped=False, showLocal=False, showOthers=False):

        self.initPlot(
            recordingId=sceneData.pedData.recordingId[0],
            title=f"Trajectories for location{sceneData.locationId} and scene {sceneData.sceneId}",
            backgroundImagePath=sceneData.backgroundImagePath
        )

        # show pedestrians
        uniqueCrossingIds = sceneData.uniquePedIds()
        for uniqueTrackId in uniqueCrossingIds:
            if not onlyClipped:
                pedDf = sceneData.getPedDfByUniqueTrackId(uniqueTrackId)
                self.plot(sceneData.orthoPxToMeter, pedDf)

            clippedDf = sceneData.getPedDfByUniqueTrackId(
                uniqueTrackId, clipped=True)
            self.plot(sceneData.orthoPxToMeter, clippedDf, style="c:")

            if showLocal:
                # localDf = sceneData.getPedDataInSceneCorrdinates()
                self.plot(sceneData.orthoPxToMeter, clippedDf,
                          xCol='sceneX', yCol='sceneY')

        # show others
        if showOthers:
            uniqueCrossingIds = sceneData.uniqueOtherIds()
            for uniqueTrackId in uniqueCrossingIds:
                if not onlyClipped:
                    otherDf = sceneData.getOtherDfByUniqueTrackId(
                        uniqueTrackId)
                    self.plot(sceneData.orthoPxToMeter, otherDf, style="w")

                clippedDf = sceneData.getOtherDfByUniqueTrackId(
                    uniqueTrackId, clipped=True)
                self.plot(sceneData.orthoPxToMeter, clippedDf, style="y:")

                if showLocal:
                    # localDf = sceneData.getPedDataInSceneCorrdinates()
                    self.plot(sceneData.orthoPxToMeter, clippedDf,
                              style="w", xCol='sceneX', yCol='sceneY')

        # plot scene bounding box
        ortho_px_to_meter = sceneData.orthoPxToMeter * self.scale_down_factor
        (X, Y) = sceneData.polygon.exterior.xy
        X = np.array(X)
        Y = np.array(Y)
        X /= ortho_px_to_meter
        Y /= -ortho_px_to_meter
        self.ax.plot(X, Y, "lime")

        # plot scene center

        X = [sceneData.centerX / ortho_px_to_meter]
        Y = [sceneData.centerY / -ortho_px_to_meter]
        self.ax.plot(X, Y, marker='o', markersize=10,
                     markerfacecolor="yellow", markeredgecolor="black")

    def showLocalTrajectories(self, df, idCol, xCol, yCol):
        pedIds = df[idCol].unique()
        for pedId in pedIds:
            pedDf = df[df[idCol] == pedId]
            plt.plot(pedDf[xCol], pedDf[yCol])
