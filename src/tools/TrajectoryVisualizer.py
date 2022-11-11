from extractors.loader import Loader
from extractors.LocationData import LocationData
from extractors.SceneCrossingData import SceneCrossingData
from loguru import logger
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

class TrajectoryVisualizer:
  """for InD only"""

  def __init__(self, loader: Loader):

    self.loader = loader
    self.scale_down_factor = 12
  
  def initPlot(self, recordingId, title):
  
    # Create figure and axes
    self.fig, self.ax = plt.subplots(1, 1)
    self.fig.set_size_inches(15, 8)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.00)
    self.fig.canvas.set_window_title(title)
    self.ax.set_title(title)

    background_image_path = self.loader.getBackgroundImagePath(recordingId)
    if background_image_path and os.path.exists(background_image_path):
        logger.info("Loading background image from {}", background_image_path)
        self.background_image = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
        (self.image_height, self.image_width) = self.background_image.shape[:2]
    else:
        logger.warning("No background image given or path not valid. Using fallback black background.")
        self.image_height, self.image_width = 1700, 1700
        self.background_image = np.zeros((self.image_height, self.image_width, 3), dtype="uint8")
    self.ax.imshow(self.background_image)
  


  def plot(self, orthoPxToMeter, tracksDf, style="r", xCol='xCenter', yCol='yCenter'):
    ortho_px_to_meter = orthoPxToMeter * self.scale_down_factor # TODO fixed to scale down factor of inD

    self.ax.plot(tracksDf[xCol] / ortho_px_to_meter, -tracksDf[yCol] / ortho_px_to_meter, style)


  def showTrack(self, tracksDf, recordingMeta, trackId):
    self.initPlot(recordingMeta["recordingId"], f"Trajectory for track {trackId}")

    pedTracksDf = tracksDf[tracksDf.trackId == trackId]
    self.plot(recordingMeta["orthoPxToMeter"], pedTracksDf)
    return pedTracksDf

  def showLocationCrossingTracks(self, locationData: LocationData):
    self.initPlot(locationData.recordingIds[0], f"Trajectories for location{locationData.locationId}")
    crossingDf = locationData.getCrossingDf()

    uniqueCrossingIds = locationData.getUniqueCrossingIds()
    for uniqueTrackId in uniqueCrossingIds:
      pedDf = locationData.getCrossingDfByUniqueTrackId(uniqueTrackId)
      self.plot(locationData.orthoPxToMeter, pedDf)

  def showLocationSceneCrossingData(self, sceneCrossingData: SceneCrossingData, onlyClipped=False, showLocal=False, showOthers=False):

    self.initPlot(sceneCrossingData.pedData.recordingId[0], f"Trajectories for location{sceneCrossingData.locationId} and scene {sceneCrossingData.sceneId}")

    # show pedestrians
    uniqueCrossingIds = sceneCrossingData.uniquePedIds()
    for uniqueTrackId in uniqueCrossingIds:
      if not onlyClipped:
        pedDf = sceneCrossingData.getPedDfByUniqueTrackId(uniqueTrackId)
        self.plot(sceneCrossingData.orthoPxToMeter, pedDf)
      
      clippedDf = sceneCrossingData.getPedDfByUniqueTrackId(uniqueTrackId, clipped=True)
      self.plot(sceneCrossingData.orthoPxToMeter, clippedDf, style="c:")

      if showLocal:
        # localDf = sceneCrossingData.getPedDataInSceneCorrdinates()
        self.plot(sceneCrossingData.orthoPxToMeter, clippedDf, xCol='sceneX', yCol='sceneY')


    # show others
    if showOthers:
      uniqueCrossingIds = sceneCrossingData.uniqueOtherIds()
      for uniqueTrackId in uniqueCrossingIds:
        if not onlyClipped:
          otherDf = sceneCrossingData.getOtherDfByUniqueTrackId(uniqueTrackId)
          self.plot(sceneCrossingData.orthoPxToMeter, otherDf, style="w")
        
        clippedDf = sceneCrossingData.getOtherDfByUniqueTrackId(uniqueTrackId, clipped=True)
        self.plot(sceneCrossingData.orthoPxToMeter, clippedDf, style="y:")

        if showLocal:
          # localDf = sceneCrossingData.getPedDataInSceneCorrdinates()
          self.plot(sceneCrossingData.orthoPxToMeter, clippedDf, style="w", xCol='sceneX', yCol='sceneY')

    
    # plot scene bounding box
    ortho_px_to_meter = sceneCrossingData.orthoPxToMeter * self.scale_down_factor
    (X, Y) = sceneCrossingData.polygon.exterior.xy
    X = np.array(X)
    Y = np.array(Y)
    X /= ortho_px_to_meter
    Y /= -ortho_px_to_meter
    self.ax.plot(X, Y, "lime")

    # plot scene center

    X = [sceneCrossingData.centerX / ortho_px_to_meter]
    Y = [sceneCrossingData.centerY/ -ortho_px_to_meter]
    self.ax.plot(X, Y, marker='o', markersize=10, markerfacecolor="yellow", markeredgecolor="black")

  
  def showLocalTrajectories(self, df, idCol, xCol, yCol):
    pedIds = df[idCol].unique()
    for pedId in pedIds:
        pedDf = df[df[idCol] == pedId]
        plt.plot(pedDf[xCol], pedDf[yCol])







  
