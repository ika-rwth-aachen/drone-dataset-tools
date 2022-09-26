import pandas as pd
from sortedcontainers import SortedList
from tools.TrajectoryUtils import TrajectoryUtils
import logging
from tqdm import tqdm

class RecordingData:

  def __init__(self, recordingId, recordingMeta, tracksMetaDf, tracksDf):

    self.recordingId = recordingId
    self.recordingMeta = recordingMeta
    self.tracksMetaDf = tracksMetaDf
    self.tracksDf = tracksDf

    self.__crossingDfByAnnotation = None
    self.__crossingDfBySceneConfig = None
    self.__sceneData = {}


  def getPedIds(self) -> SortedList:
    return SortedList(self.tracksMetaDf[self.tracksMetaDf['class'] == 'pedestrian']['trackId'].tolist())
  

  def getCrossingPedIds(self) -> SortedList:
    if 'crossing' not in self.tracksMetaDf:
      raise Exception("crossing annotation not in tracksMetaDf")

    return SortedList(self.tracksMetaDf[(self.tracksMetaDf['class'] == 'pedestrian') & (self.tracksMetaDf['crossing'] == 'yes') ]['trackId'].tolist())


  def getDfByTrackIds(self, trackIds):
      criterion = self.tracksDf['trackId'].map(lambda trackId: trackId in trackIds)
      return self.tracksDf[criterion]

  
  def getCrossingDfByAnnotations(self):
    if self.__crossingDfByAnnotation is None:
      crossingIds = self.getCrossingPedIds()
      self.__crossingDfByAnnotation = self.getDfByTrackIds(crossingIds)

    return self.__crossingDfByAnnotation


  def getCrossingDfBySceneConfig(self, sceneConfigs, refresh=False, fps=2.5):
      logging.debug(f"getCrossingDfBySceneConfig from recording {self.recordingId}")
    
      if self.__crossingDfBySceneConfig is None:
        sceneIds = list(sceneConfigs.keys())
        sceneDfs = []
        # 1. Get all clipped scene crossing data in their local coordinate system
        for sceneId  in sceneIds:
          logging.info(f"extracting crossing data for scene {sceneId} from recording {self.recordingId}")
          sC = sceneConfigs[str(sceneId)]
          df = self.getCrossingDfForScene(sceneId, sC, refresh=False, fps=2.5)
          df["sceneId"] = sceneId
          sceneDfs.append(df)
        
        self.__crossingDfBySceneConfig = pd.concat(sceneDfs, ignore_index=True)

      return self.__crossingDfBySceneConfig


  def getSceneData(self, sceneId, sceneConfig, refresh=False, fps=2.5):
    
    if sceneId not in self.__sceneData or refresh:

      data = self.getCrossingDfForScene(sceneId, sceneConfig["boxWidth"], sceneConfig["roadWidth"], refresh=False, fps=2.5)
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

  def getCrossingDfForScene(self, sceneId, sceneConfig, refresh=False, fps=2.5) -> pd.DataFrame:

    sceneDfs = []

    # create polygon
    scenePolygon = TrajectoryUtils.scenePolygon(sceneConfig, sceneConfig["boxWidth"], sceneConfig["roadWidth"] / 2)
    # create splines
    pedIds = self.getPedIds()
    for pedId in tqdm(pedIds, desc="pedIds", leave=True, position=0):
      pedDf = self.getDfByTrackIds([pedId])
      trajSpline = TrajectoryUtils.dfToSplines(pedDf, "xCenter", "yCenter", 1)
      if TrajectoryUtils.doesIntersect(scenePolygon, trajSpline):
        pedDf = pedDf.copy() # we can modify without concern now
        pedDf["sceneId"] = sceneId
        sceneDfs.append(pedDf)
    
    return pd.concat(sceneDfs, ignore_index=True)



  