import pandas as pd
import numpy as np
from shapely.geometry import Point
from tools.TrajectoryUtils import TrajectoryUtils
from loguru import logger
from tqdm import tqdm

class SceneCrossingData:
  """SceneCrossingData only has crossing trajectories.
  """
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

    self._dropWorldCoordinateColumns()
    self._transformToLocalCoordinate()


  def uniquePedIds(self) -> np.ndarray:
    if self._pedIds is None:
      self._pedIds = self.data.uniqueTrackId.unique()
    
    return self._pedIds

  def uniqueClippedPedIds(self) -> np.ndarray:
      clippedDf = self.getClippedDfs()
      if len(clippedDf) > 0:
        return clippedDf.uniqueTrackId.unique()
      return []


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
  
  def _dropWorldCoordinateColumns(self):
      logger.debug("Dropping , lonVelocity, latVelocity, lonAcceleration, latAcceleration")
      self.data = self.data.drop(["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"], axis=1)
  
  def _transformToLocalCoordinate(self):
      logger.debug("transforming trajectories to scene coordinates")

      # translate and rotate.
      clippedDf = self.getClippedDfs()
      
      # transform position
      # transform velocity
      # transform heading

      origin = Point(self.centerX, self.centerY)
      originAngle = self.angle

      translationMat = TrajectoryUtils.getTranslationMatrix(origin)
      rotationMat = TrajectoryUtils.getRotationMatrix(originAngle)

      sceneX = []
      sceneY = []


      for idx, row in clippedDf.iterrows():

        position = Point(row["xCenter"], row["yCenter"])
        # velocity = (row["xVelocity"], row["yVelocity"])
        # acceleration = (row["xAcceleration"], row["yAcceleration"])
        # heading = row['heading']
        newPosition = TrajectoryUtils.transformPoint(translationMat, rotationMat, position)

        # row['sceneX'] = newPosition.x
        # row['sceneY'] = newPosition.y
        sceneX.append(newPosition.x)
        sceneY.append(newPosition.y)

      
      clippedDf["sceneX"] = sceneX
      clippedDf["sceneY"] = sceneY
      self._dataLocal = clippedDf

      pass
  
  def getDataInSceneCorrdinates(self):
    if self._dataLocal is None:
      self.transformToLocalCoordinate()
    
    return self._dataLocal
  
  

  
  # def _clip(self):
  #   logger.debug("clipping trajectories")
  #   dfs = []
  #   for pedId in  tqdm(self.uniquePedIds(), desc="clipping trajectories"):
  #     pedDf = self.getDfByUniqueTrackId(pedId)
  #     clippedDf = TrajectoryUtils.clip(pedDf, "xCenter", "yCenter", "frame", self.sceneConfig, self.sceneConfig["boxWidth"], self.sceneConfig["roadWidth"] + 2)
  #     if TrajectoryUtils.length(clippedDf, "xCenter", "yCenter") < self.sceneConfig["roadWidth"]:
  #       logger.debug(f"Disregarding trajectory for {pedId} because the length is too low")
  #     else:
  #       dfs.append(clippedDf)
    
  #   self._clippedData = pd.concat(dfs, ignore_index=True)
  
  
  def _clip(self):
    logger.debug("clipping trajectories")
    scenePolygon = TrajectoryUtils.scenePolygon(self.sceneConfig, self.sceneConfig["boxWidth"], self.sceneConfig["roadWidth"] + 2)
    dfs = []
    for pedId in  tqdm(self.uniquePedIds(), desc=f"clipping trajectories for scene # {self.sceneId}"):
      pedDf = self.getDfByUniqueTrackId(pedId)
      clippedDf = TrajectoryUtils.clipByRect(pedDf, "xCenter", "yCenter", "frame", scenePolygon)
      if TrajectoryUtils.length(clippedDf, "xCenter", "yCenter") < self.sceneConfig["roadWidth"]:
        logger.debug(f"Disregarding trajectory for {pedId} because the length is too low")
      else:
        dfs.append(clippedDf)

    if len(dfs) == 0:
      """No data"""
      self._clippedData = pd.DataFrame()
    else:
      self._clippedData = pd.concat(dfs, ignore_index=True)

  
  def getClippedDfs(self):
    if self._clippedData is None:
      self._clip()
    
    return self._clippedData

  
  def clippedSize(self):
    return len(self.uniqueClippedPedIds())
