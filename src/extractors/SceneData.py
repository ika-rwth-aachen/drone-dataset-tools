import pandas as pd
import numpy as np
from shapely.geometry import Point
from tools.TrajectoryUtils import TrajectoryUtils

class SceneData:
  def __init__(
    self, 
    locationId, 
    orthoPxToMeter,
    sceneId, 
    sceneConfig, 
    boxWidth, 
    boxHeight, 
    data: pd.DataFrame
    ):

    self.locationId = locationId
    self.orthoPxToMeter = orthoPxToMeter # for visualization
    self.sceneId = sceneId
    self.sceneConfig = sceneConfig
    self.centerX = sceneConfig["centerX"]
    self.centerY = sceneConfig["centerY"]
    self.angle = sceneConfig["angle"]

    self.boxWidth = boxWidth
    self.boxHeight = boxHeight
    self.polygon = TrajectoryUtils.scenePolygon(sceneConfig, boxWidth, boxHeight)

    self.data = data
    self._clippedData = None
    self._dataLocal = None

    self._pedIds = None

    self.dropWorldCoordinateColumns()

  def uniquePedIds(self) -> np.ndarray:
    if self._pedIds is None:
      self._pedIds = self.data.uniqueTrackId.unique()
    
    return self._pedIds


  def getDfByUniqueTrackId(self, uniqueTrackId, clipped=False):
    return self.getDfByUniqueTrackIds([uniqueTrackId], clipped=clipped)

  def getDfByUniqueTrackIds(self, uniqueTrackIds, clipped=False):
      
      if clipped:
        clippedDf = self.getClippedDfs()
        criterion = clippedDf['uniqueTrackId'].map(lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
        return clippedDf[criterion]
      else:
        criterion = self.data['uniqueTrackId'].map(lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
        return self.data[criterion]
  
  def transformToLocalCoordinate(self):

      # translate and rotate.
      clippedDf = self.getClippedDfs().clone()
      
      # transform position
      # transform velocity
      # transform heading

      origin = Point(self.centerX, self.centerY)
      originAngle = self.angle


      for idx, row in clippedDf.iterrows():

        position = (row["xCenter"], row["yCenter"])
        velocity = (row["xVelocity"], row["yVelocity"])
        acceleration = (row["xAcceleration"], row["yAcceleration"])
        heading = row['heading']


      pass
  
  
  def dropWorldCoordinateColumns(self):
    self.data = self.data.drop(["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1)

  
  def clip(self):
    dfs = []
    for pedId in self.uniquePedIds():
      pedDf = self.getDfByUniqueTrackId(pedId)
      clippedDf = TrajectoryUtils.clip(pedDf, "xCenter", "yCenter", "frame", self.sceneConfig, self.sceneConfig["boxWidth"], self.sceneConfig["roadWidth"] + 2)
      if TrajectoryUtils.length(clippedDf, "xCenter", "yCenter") < self.sceneConfig["roadWidth"]:
        print(f"Disregarding trajectory for {pedId} because the length is too low")
      else:
        dfs.append(clippedDf)
    
    self._clippedData = pd.concat(dfs, ignore_index=True)
  
  def getClippedDfs(self):
    if self._clippedData is None:
      self.clip()
    
    return self._clippedData
