import pandas as pd
from sortedcontainers import SortedList
from loguru import logger
from tools.UnitUtils import UnitUtils
from tools.TrajectoryUtils import TrajectoryUtils
from .SceneData import SceneData
from tqdm import tqdm
from dill import dump, load
import os
import functools

class LocationData:

  def __init__(self, locationId, recordingIds, recordingDataList, useSceneConfigToExtract=False, precomputeSceneData=True):


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

    self._mergedSceneDfs = {}

    if precomputeSceneData:
      self._precomputeSceneData()

  

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

  
  def summary(self):

    summary =  {
      "#original frameRate": self.frameRate,
      "#crossing trajectories": len(self.getUniqueCrossingIds()),
      "#scene trajectories": functools.reduce(lambda acc, new: acc + new, [sceneData.clippedSize() for sceneData in self.__sceneData.values()])
    }

    for sceneId in self.__sceneData.keys():
      summary[f"scene#{sceneId}"] = self.__sceneData[sceneId].clippedSize()

    return summary
  
  #region extraction
  def getUniqueCrossingIds(self):
    """
    returns unique pedestrian ids
    """

    if self.__crossingIds is None:
      self.__crossingIds = SortedList()
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
          logger.info(f"got crossing df for {recordingData.recordingId}")
          if "uniqueTrackId" not in crossingDf:
            raise Exception(f"{recordingData.recordingId} does not have uniqueTrackId")
          dfs.append(crossingDf)
        except Exception as e:
          logger.warning(f"{recordingData.recordingId} has exception: {e}")
          # raise e

      self.__crossingDf = pd.concat(dfs, ignore_index=True)
    
    return self.__crossingDf
  #endregion

  #region scene
  def _precomputeSceneData(self):
    sceneConfigs = self.getSceneConfig()
    sceneIds = list(sceneConfigs.keys())

    for sceneId in sceneIds:
      sceneConfig = sceneConfigs[sceneId]
      self.getSceneCrossingData(sceneId, sceneConfig["boxWidth"], sceneConfig["roadWidth"])

    

  def getSceneConfig(self):
    allLocationSceneConfig = UnitUtils.loadSceneConfiguration()
    return allLocationSceneConfig[str(self.locationId)]

  
  def getSceneCrossingDf(self, sceneId, boxWidth, boxHeight) -> pd.DataFrame:

    sceneId = str(sceneId)
    if self.useSceneConfigToExtract:
      crossingDf = self.getCrossingDf()
      return crossingDf[crossingDf["sceneId"] == sceneId].copy().reset_index()

    
    logger.info(f"collecting scene crossing data from annotated data for scene {sceneId}")

    sceneDfs = []
    sceneConfig = self.getSceneConfig()[sceneId]

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

  
  def getSceneCrossingData(self, sceneId, boxWidth=6, boxHeight=6, refresh=False, fps=2.5) -> SceneData:
    """_summary_

    Args:
        sceneId (_type_): scene id from scene config file
        boxWidth (int, optional): width of the bounding box to filter unrelated trajectories. Runs along the road's length. Defaults to 6.
        boxHeight (int, optional): height of the bounding box to filter unrelated trajectories. Runs along the road's width. Defaults to 6.
        refresh (bool, optional): force re-filter when bounding box changes. Results are cached when run. Defaults to False.
        fps (float, optional): frame rate conversion from 25. Defaults to 2.5.

    Returns:
        SceneData: _description_
    """

    sceneId = str(sceneId)
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

  
  def mergeScenesByRoadWidth(self):
    """
      merges local coordinates
    """

    if len(self._mergedSceneDfs ) > 0:
      return self._mergedSceneDfs

    sceneConfigs = self.getSceneConfig()
    sceneIds = list(sceneConfigs.keys())

    groups = {}
    for sceneId in sceneIds:
      sceneConfig = sceneConfigs[sceneId]
      if sceneConfig["roadWidth"] not in groups:
        groups[sceneConfig["roadWidth"]] = []
      groups[sceneConfig["roadWidth"]].append(self.getSceneCrossingData(sceneId, 0, 0))

    for roadWidth in tqdm(groups, desc="merging scenes"):
      groupDfs = []
      group = groups[roadWidth]
      for sceneData in group:
        sceneLocalDf = sceneData.getDataInSceneCorrdinates()
        groupDfs.append(sceneLocalDf[["frame", "uniqueTrackId", "sceneX", "sceneY", "recordingId"]].copy())
      groupDf = pd.concat(groupDfs, ignore_index=True)
      groupDf["roadWidth"] = roadWidth
      self._mergedSceneDfs[roadWidth] = groupDf

    return self._mergedSceneDfs



  #endregion

  #region third party formats  
  def getCrossingDataForTransformerNoMerge(self):
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
      if len(sceneLocalDf) > 0:
        sceneDfs.append(sceneLocalDf[["frame", "uniqueTrackId", "sceneX", "sceneY", "sceneId", "recordingId"]].copy())

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

  def getCrossingDataForTransformer(self):
    allSceneDfs = {}
    for roadWidth in self._mergedSceneDfs:
      allSceneDfs[roadWidth] = self._mergedSceneDfs[roadWidth].copy().rename(columns={
                                                                              "uniqueTrackId": "ped", 
                                                                              "sceneX": "x", 
                                                                              "sceneY": "y"
                                                                            })
    return allSceneDf


  #endregion

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
      logger.info(f"saved to {fpath}")
    
  @staticmethod
  def load(locDir):
    fpath = os.path.join(locDir, "all.dill")
    logger.info(f"reading from {fpath}")
    with open(fpath, "rb") as fp:
      return load(fp)
  #endregion









