import pandas as pd
import numpy as np
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
    self.centerX = sceneConfig["centerX"]
    self.centerY = sceneConfig["centerY"]
    self.angle = sceneConfig["angle"]

    self.boxWidth = boxWidth
    self.boxHeight = boxHeight
    self.polygon = TrajectoryUtils.scenePolygon(sceneConfig, boxWidth, boxHeight)

    self.data = data

    self._pedIds = None

  def uniquePedIds(self) -> np.ndarray:
    if self._pedIds is None:
      self._pedIds = self.data.uniqueTrackId.unique()
    
    return self._pedIds


  def getDfByUniqueTrackId(self, uniqueTrackId):
    return self.getDfByUniqueTrackIds([uniqueTrackId])

  def getDfByUniqueTrackIds(self, uniqueTrackIds):
      criterion = self.data['uniqueTrackId'].map(lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
      return self.data[criterion]