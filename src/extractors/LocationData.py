import pandas as pd
from sortedcontainers import SortedList
import logging
from tools.UnitUtils import UnitUtils
from tools.TrajectoryUtils import TrajectoryUtils
from .SceneData import SceneData
from tqdm import tqdm
from dill import dump, load
import os

class LocationData:

  def __init__(self, locationId, recordingIds, recordingDataList, useSceneConfigToExtract=False):


    self.locationId = locationId
    self.recordingIds = recordingIds
    self.recordingDataList = recordingDataList

    self.useSceneConfigToExtract = useSceneConfigToExtract

    self.recordingMetaList = [recordingData.recordingMeta for recordingData in recordingDataList]
    self.validateRecordingMeta()

    self.frameRate = self.recordingMetaList[0]["frameRate"]
    self.orthoPxToMeter = self.recordingMetaList[0]["orthoPxToMeter"]
  
    # cache
    self.__crossingDf = None
    self.__crossingIds = None

    self.__sceneData = {}

  

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
          raise Exception(f"{field} value mismatch for {firstMeta['recordingId']} and {otherMeta['recordingId']}")

  def getUniqueCrossingIds(self):
    """
    returns unique pedestrian ids
    """

    if self.__crossingIds is None:
      self.__crossingIds = SortedList()
      # for recordingData in self.recordingDataList:
      #   # crossingIds = recordingData.getCrossingIds()
      #   try:
          
      #     crossingDf = recordingData.getCrossingDf()
      #     if "uniqueTrackId" in crossingDf:
      #       uniqueIds = crossingDf.uniqueTrackId.unique()
      #       logging.info(f"crossing ids for {recordingData.recordingId}: {recordingData.getCrossingPedIds()}")
      #       logging.info(f"uniqueIds for {recordingData.recordingId}: {uniqueIds}")
      #       self.__crossingIds.update(uniqueIds)
      #     else:
      #       logging.warn(f"{recordingData.recordingId} does not have uniqueTrackId")
      #   except Exception as e:
      #     logging.warn(f"{recordingData.recordingId} has exception: {e}")
      crossingDf = self.getCrossingDf()
      self.__crossingIds.update(crossingDf.uniqueTrackId.unique())

    return self.__crossingIds

  def getCrossingDfByUniqueTrackId(self, uniqueTrackId):
    return self.getCrossingDfByUniqueTrackIds([uniqueTrackId])

  def getCrossingDfByUniqueTrackIds(self, uniqueTrackIds):
      crossingDf = self.getCrossingDf()
      criterion = crossingDf['uniqueTrackId'].map(lambda uniqueTrackId: uniqueTrackId in uniqueTrackIds)
      return crossingDf[criterion]

  def getRecordingCrossingDf(self, recordingData):
      if self.useSceneConfigToExtract:
        crossingDf = recordingData.getCrossingDfBySceneConfig(self.getSceneConfig())
      else:
        crossingDf = recordingData.getCrossingDfByAnnotations()
      return crossingDf

  def getCrossingDf(self):
    if self.__crossingDf is None:
      dfs = []
      for recordingData in tqdm(self.recordingDataList, desc="recording", position=0):
        try:
          crossingDf = self.getRecordingCrossingDf(recordingData)
          logging.info(f"got crossing df for {recordingData.recordingId}")
          if "uniqueTrackId" not in crossingDf:
            raise Exception(f"{recordingData.recordingId} does not have uniqueTrackId")
          dfs.append(crossingDf)
        except Exception as e:
          logging.warn(f"{recordingData.recordingId} has exception: {e}")
          # raise e

      self.__crossingDf = pd.concat(dfs, ignore_index=True)
    
    return self.__crossingDf

  
  def getSceneConfig(self):
    allLocationSceneConfig = UnitUtils.loadSceneConfiguration()
    return allLocationSceneConfig[str(self.locationId)]

  
  def getSceneCrossingDf(self, sceneId, boxWidth, boxHeight) -> pd.DataFrame:

    if self.useSceneConfigToExtract:
      crossingDf = self.getCrossingDf()
      return crossingDf[crossingDf["sceneId"] == str(sceneId)].copy().reset_index()

    sceneDfs = []
    sceneConfig = self.getSceneConfig()[str(sceneId)]

    # create polygon
    scenePolygon = TrajectoryUtils.scenePolygon(sceneConfig, boxWidth, boxHeight)
    # create splines
    crossingIds = self.getUniqueCrossingIds()
    for crossingId in crossingIds:
      pedDf = self.getCrossingDfByUniqueTrackId(crossingId)
      trajSpline = TrajectoryUtils.dfToSplines(pedDf, "xCenter", "yCenter", 1)
      if TrajectoryUtils.doesIntersect(scenePolygon, trajSpline):
        pedDf = pedDf.copy() # we can modify without concern now
        pedDf["sceneId"] = sceneId
        sceneDfs.append(pedDf)
    
    return pd.concat(sceneDfs, ignore_index=True)

  
  def getSceneCrossingData(self, sceneId, boxWidth, boxHeight, refresh=False, fps=2.5) -> SceneData:

    if sceneId not in self.__sceneData or refresh:

      data = self.getSceneCrossingDf(sceneId, boxWidth, boxHeight)
      sceneConfig = self.getSceneConfig()[str(sceneId)]
      self.__sceneData[sceneId] = SceneData(
                                            self.locationId, 
                                            self.orthoPxToMeter,
                                            sceneId, 
                                            sceneConfig, 
                                            boxWidth, 
                                            boxHeight, 
                                            data
                                          )

    return self.__sceneData[sceneId]

  
  def getCrossingDataForTransformer(self):
    """
    csv with frame, ped, x, y
    """
    sceneConfigs = self.getSceneConfig()
    sceneIds = list(sceneConfigs.keys())

    sceneDfs = []

    # 1. Get all clipped scene crossing data in their local coordinate system
    for sceneId  in sceneIds:
      sceneConfig = sceneConfigs[str(sceneId)]
      sceneData = self.getSceneCrossingData(sceneId, sceneConfig["boxWidth"], sceneConfig["roadWidth"])
      sceneLocalDf = sceneData.getDataInSceneCorrdinates()
      sceneDfs.append(sceneLocalDf[["frame", "uniqueTrackId", "sceneX", "sceneY", "sceneId"]].copy())

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

  #region cache
  def madeLocationDir(self, outputDir):
    locDir = os.path.join(outputDir, f"location-{self.locationId}")
    os.makedirs(locDir, exist_ok = True)
    return locDir


  def saveCrossingDf(self, outputDir):

    locDir = self.madeLocationDir(outputDir)
    fpath = os.path.join(locDir, "crossing.csv")
    crossingDf = self.getCrossingDf()
    crossingDf.to_csv(fpath)

  
  def save(self, outputDir):

    locDir = self.madeLocationDir(outputDir)
    fpath = os.path.join(locDir, "all.dill")
    with open(fpath, "wb") as fp:
      dump(self, fp)
      logging.info(f"saved to {fpath}")
    
  @staticmethod
  def load(locDir):
    fpath = os.path.join(locDir, "all.dill")
    logging.info(f"reading from {fpath}")
    with open(fpath, "rb") as fp:
      return load(fp)
  #endregion









