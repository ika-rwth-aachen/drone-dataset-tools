import pandas as pd
from sortedcontainers import SortedList
from loguru import logger
from tools.UnitUtils import UnitUtils
from tools.TrajectoryUtils import TrajectoryUtils
from .SceneCrossingData import SceneCrossingData
from tqdm import tqdm
from dill import dump, load
from datetime import datetime
import os
import functools

class LocationData:

  def __init__(self, locationId, recordingIds, recordingDataList, useSceneConfigToExtract=False, precomputeSceneCrossingData=True):
    """_summary_

    Args:
        locationId (_type_): _description_
        recordingIds (_type_): _description_
        recordingDataList (_type_): _description_
        useSceneConfigToExtract (bool, optional): We extract data in two ways:. Defaults to False.
        precomputeSceneCrossingData (bool, optional): extracts data. Defaults to True.
    """


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

    self.__otherDf = None
    self.__otherIds = None

    self.__SceneCrossingData = {}

    self._mergedSceneDfs = {}

    if precomputeSceneCrossingData:
      self._precomputeSceneCrossingData()

  

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
      "#scene trajectories": functools.reduce(lambda acc, new: acc + new, [SceneCrossingData.clippedPedSize() for SceneCrossingData in self.__SceneCrossingData.values()])
    }

    for sceneId in self.__SceneCrossingData.keys():
      summary[f"scene#{sceneId}"] = self.__SceneCrossingData[sceneId].clippedPedSize()

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
        crossingDf = recordingData.getCrossingDfByAnnotations() # it does not have scene id!
      return crossingDf

  def getRecordingOtherDf(self, recordingData):
      return recordingData.getOtherDfBySceneConfig(self.getSceneConfig())

  def getCrossingDf(self):
    """returns crossing data for all the scenes and recordings

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if self.__crossingDf is None:
      crossingDfs = []
      for recordingData in tqdm(self.recordingDataList, desc="crossing recording", position=0):
        try:
          crossingDf = self.getRecordingCrossingDf(recordingData)
          logger.info(f"got crossing df for {recordingData.recordingId}")
          if "uniqueTrackId" not in crossingDf:
            raise Exception(f"{recordingData.recordingId} does not have uniqueTrackId")
          crossingDfs.append(crossingDf)
        except Exception as e:
          logger.warning(f"{recordingData.recordingId} has exception: {e}")
          # raise e

      self.__crossingDf = pd.concat(crossingDfs, ignore_index=True)
    
    return self.__crossingDf

  def getOtherDf(self):
    """returns other data for all the scenes and recordings

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if self.__otherDf is None:
      otherDfs = []
      for recordingData in tqdm(self.recordingDataList, desc="other recording", position=0):
        try:
          otherDf = self.getRecordingOtherDf(recordingData)
          logger.info(f"got other df for {recordingData.recordingId}")
          if "uniqueTrackId" not in otherDf:
            raise Exception(f"{recordingData.recordingId} does not have uniqueTrackId")
          otherDfs.append(otherDf)
        except Exception as e:
          logger.warning(f"{recordingData.recordingId} has exception: {e}")
          # raise e

      self.__otherDf = pd.concat(otherDfs, ignore_index=True)
    
    return self.__otherDf
  #endregion

  #region scene
  def _precomputeSceneCrossingData(self):
    sceneConfigs = self.getSceneConfig()
    sceneIds = list(sceneConfigs.keys())

    for sceneId in sceneIds:
      logger.info(f"Precomputing sceneCrossingData for {sceneId}")
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

    
  def getSceneOtherDf(self, sceneId) -> pd.DataFrame:

    sceneId = str(sceneId)
    otherDf = self.getOtherDf()
    return otherDf[otherDf["sceneId"] == sceneId].copy().reset_index()

  
  def getSceneCrossingData(self, sceneId, boxWidth=6, boxHeight=6, refresh=False, fps=2.5) -> SceneCrossingData:
    """_summary_

    Args:
        sceneId (_type_): scene id from scene config file
        boxWidth (int, optional): width of the bounding box to filter unrelated trajectories. Runs along the road's length. Defaults to 6.
        boxHeight (int, optional): height of the bounding box to filter unrelated trajectories. Runs along the road's width. Defaults to 6.
        refresh (bool, optional): force re-filter when bounding box changes. Results are cached when run. Defaults to False.
        fps (float, optional): frame rate conversion from 25. Defaults to 2.5.

    Returns:
        SceneCrossingData: _description_
    """

    sceneId = str(sceneId)
    if sceneId not in self.__SceneCrossingData or refresh:

      otherData = self.getSceneOtherDf(sceneId)
      pedData = self.getSceneCrossingDf(sceneId, boxWidth, boxHeight)
      sceneConfig = self.getSceneConfig()[str(sceneId)]
      self.__SceneCrossingData[sceneId] = SceneCrossingData(
                                            self.locationId, 
                                            self.orthoPxToMeter,
                                            sceneId, 
                                            sceneConfig, 
                                            boxWidth, 
                                            boxHeight, 
                                            pedData=pedData,
                                            otherData=otherData
                                          )

    return self.__SceneCrossingData[sceneId]

  
  def mergeScenesByRoadWidth(self, refresh=False):
    """
      merges local coordinates
    """

    if len(self._mergedSceneDfs ) > 0 and not refresh:
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
      for SceneCrossingData in group:
        sceneLocalDf = SceneCrossingData.getPedDataInSceneCorrdinates()
        groupDfs.append(sceneLocalDf[["frame", "uniqueTrackId", "sceneX", "sceneY", "sceneId", "recordingId"]].copy())
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
      SceneCrossingData = self.getSceneCrossingData(sceneId, sceneConfig["boxWidth"], sceneConfig["roadWidth"])
      sceneLocalDf = SceneCrossingData.getPedDataInSceneCorrdinates()
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

  def getCrossingDataForTransformer(self, refresh=False):
    allSceneDfs = {}
    mergedSceneDfs = self.mergeScenesByRoadWidth(refresh=refresh)
    for roadWidth in mergedSceneDfs:
      allSceneDfs[roadWidth] = mergedSceneDfs[roadWidth].copy().rename(columns={
                                                                              "uniqueTrackId": "ped", 
                                                                              "sceneX": "x", 
                                                                              "sceneY": "y"
                                                                            })
    return allSceneDfs


  #endregion

  #region cache
  def madeLocationDir(self, outputDir):
    locDir = os.path.join(outputDir, f"location-{self.locationId}")
    os.makedirs(locDir, exist_ok = True)
    return locDir


  def saveCrossingDf(self, outputDir):

    locDir = self.madeLocationDir(outputDir)
    date_time = datetime.now().strftime("%Y-%m-%d")

    fpath = os.path.join(locDir, f"{date_time}-crossing.csv")
    if os.path.exists(fpath):
      os.remove(fpath)
    crossingDf = self.getCrossingDf()
    crossingDf.to_csv(fpath)

    
    fpath = os.path.join(locDir, f"{date_time}-other.csv")
    if os.path.exists(fpath):
      os.remove(fpath)
    otherDf = self.getOtherDf()
    otherDf.to_csv(fpath)

    pass

  
  def save(self, outputDir):

    locDir = self.madeLocationDir(outputDir)
    date_time = datetime.now().strftime("%Y-%m-%d")

    fpath = os.path.join(locDir, f"{date_time}-all.dill")
    if os.path.exists(fpath):
      os.remove(fpath)
    with open(fpath, "wb") as fp:
      dump(self, fp)
      logger.info(f"saved to {fpath}")
    
  @staticmethod
  def load(locDir, fname=None):

    if fname is None:
      fname = "all.dill"
      
    fpath = os.path.join(locDir, "all.dill")
    logger.info(f"reading from {fpath}")
    with open(fpath, "rb") as fp:
      return load(fp)
  #endregion









