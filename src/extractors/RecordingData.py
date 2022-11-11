import pandas as pd
from sortedcontainers import SortedList
from tools.TrajectoryUtils import TrajectoryUtils
from .SceneCrossingData import SceneCrossingData
from .TrackClass import TrackClass
from loguru import logger
from tqdm import tqdm

class RecordingData:

  def __init__(
      self, 
      recordingId, 
      recordingMeta, 
      tracksMetaDf, 
      tracksDf
    ):

    self.recordingId = recordingId
    self.recordingMeta = recordingMeta
    self.tracksMetaDf = tracksMetaDf
    self.tracksDf = tracksDf

    self.__crossingDfByAnnotation = None
    self.__crossingDfBySceneConfig = None
    self.__SceneCrossingData = {}

    self.__trackIdClassMap = {}
    self.__extractTrackIdClasses()

  @property
  def locationId(self):
    return self.recordingMeta["locationId"]

  @property
  def orthoPxToMeter(self):
    return self.recordingMeta["orthoPxToMeter"]

  def __extractTrackIdClasses(self):
    for _, row in self.tracksMetaDf.iterrows():
      self.__trackIdClassMap[row["trackId"]] = row["class"]
    pass

  def getClass(self, trackId):
    return self.__trackIdClassMap[trackId]

  def getDfByTrackIds(self, trackIds):
      criterion = self.tracksDf['trackId'].map(lambda trackId: trackId in trackIds)
      return self.tracksDf[criterion]

  
  def getIdsByClass(self, cls) -> SortedList:
    return SortedList(self.tracksMetaDf[self.tracksMetaDf['class'] == cls]['trackId'].tolist())

  def getPedIds(self) -> SortedList:
    return self.getIdsByClass(TrackClass.Pedestrian.value)

  def getCarIds(self) -> SortedList:
    return self.getIdsByClass(TrackClass.Car.value)
  
  def getBicycleIds(self) -> SortedList:
    return self.getIdsByClass(TrackClass.Bicycle.value)
  
  def getLargeVehicleIds(self) -> SortedList:
    return self.getIdsByClass(TrackClass.Truck_Bus.value)
  
  def getVehicleIds(self) -> SortedList:
    return self.getCarIds() + self.getLargeVehicleIds()

  def getDfByFrameSpan(self, start, end):
    return self.tracksDf[(self.tracksDf["frame"] >= start) & (self.tracksDf["frame"] <= end)]
  
  def getDfById(self, id):
    return self.tracksDf[self.tracksDf["trackId"] == id]

  def getDfByPedFrameSpan(self, pedId):
    """It returns all the rows that appears between the start and the end frame of the pedestrian. We probably should not use this as this does not reflect the crossing span.

    Args:
        pedId (_type_): _description_
    """
    # 1. get meta
    pedMeta = self.tracksMetaDf[self.tracksMetaDf['trackId'] == pedId]
    raise NotImplementedError("getDfByPedFrameSpan")
  

  def _getCrossingPedIdsByAnnotation(self) -> SortedList:
    if 'crossing' not in self.tracksMetaDf:
      raise Exception("crossing annotation not in tracksMetaDf")

    return SortedList(self.tracksMetaDf[(self.tracksMetaDf['class'] == 'pedestrian') & (self.tracksMetaDf['crossing'] == 'yes') ]['trackId'].tolist())

  def getCrossingDfByAnnotations(self):
    if self.__crossingDfByAnnotation is None:
      crossingIds = self._getCrossingPedIdsByAnnotation()
      self.__crossingDfByAnnotation = self.getDfByTrackIds(crossingIds)

    return self.__crossingDfByAnnotation


  def getCrossingDfBySceneConfig(self, sceneConfigs, refresh=False, fps=2.5):
      logger.debug(f"getCrossingDfBySceneConfig from recording {self.recordingId}")
    
      if self.__crossingDfBySceneConfig is None:
        sceneIds = list(sceneConfigs.keys())
        sceneDfs = []
        # 1. Get all clipped scene crossing data in their local coordinate system
        for sceneId  in sceneIds:
          logger.info(f"extracting crossing data for scene {sceneId} from recording {self.recordingId}")
          sC = sceneConfigs[str(sceneId)]
          df = self.getCrossingDfForScene(sceneId, sC, refresh=False, fps=2.5)
          if len(df) > 0:
            df["sceneId"] = sceneId
            sceneDfs.append(df)
        
        if len(sceneDfs) > 0:
          self.__crossingDfBySceneConfig = pd.concat(sceneDfs, ignore_index=True)
        else:
          logger.warning(f"No crossing data found for recording {self.recordingId}")
          self.__crossingDfBySceneConfig = pd.DataFrame()

      return self.__crossingDfBySceneConfig


  def getSceneCrossingData(self, sceneId, sceneConfig, refresh=False, fps=2.5):

    """Do not use except for fast exploration. It's not used by the LocationData extractors. Always extracts by scene config

    Returns:
        _type_: _description_
    """
    
    if sceneId not in self.__SceneCrossingData or refresh:

      data = self.getCrossingDfForScene(sceneId, sceneConfig, refresh=False, fps=2.5)
      self.__SceneCrossingData[sceneId] = SceneCrossingData(
                                            self.locationId, 
                                            self.orthoPxToMeter,
                                            sceneId, 
                                            sceneConfig, 
                                            sceneConfig["boxWidth"], 
                                            sceneConfig["roadWidth"],
                                            data
                                          )

    return self.__SceneCrossingData[sceneId]

  def getCrossingDfForScene(self, sceneId, sceneConfig, refresh=False, fps=2.5) -> pd.DataFrame:

    sceneDfs = []

    # create polygon
    scenePolygon = TrajectoryUtils.scenePolygon(sceneConfig, sceneConfig["boxWidth"], sceneConfig["roadWidth"] / 2)
    # create splines
    pedIds = self.getPedIds()
    for pedId in tqdm(pedIds, desc="pedIds", leave=True, position=0):
      pedDf = self.getDfByTrackIds([pedId])
      # trajSpline = TrajectoryUtils.dfToSplines(pedDf, "xCenter", "yCenter", 1)
      # if TrajectoryUtils.doesIntersect(scenePolygon, trajSpline):
      #   pedDf = pedDf.copy() # we can modify without concern now
      #   pedDf["sceneId"] = sceneId
      #   pedDf["roadWidth"] = sceneConfig["roadWidth"]
      #   sceneDfs.append(pedDf)
      df  = TrajectoryUtils.getDfIfDfIntersect(sceneId=sceneId, sceneConfig=sceneConfig, scenePolygon=scenePolygon, df=pedDf)
      if df is not None:
        sceneDfs.append(df)
    
    if len(sceneDfs) > 0:
      return pd.concat(sceneDfs, ignore_index=True)
    return pd.DataFrame()



  