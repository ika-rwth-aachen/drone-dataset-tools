import pandas as pd
from sortedcontainers import SortedList
import logging

class LocationData:

  def __init__(self, locationId, recordingIds, recordingDataList):
    self.locationId = locationId
    self.recordingIds = recordingIds
    self.recordingDataList = recordingDataList
  
    # cache
    self.__crossingDf = None
    self.__crossingIds = None

  

  def getUniqueCrossingIds(self):
    """
    returns unique pedestrian ids
    """

    if self.__crossingIds is None:
      self.__crossingIds = SortedList()
      for recordingData in self.recordingDataList:
        # crossingIds = recordingData.getCrossingIds()
        try:
          crossingDf = recordingData.getCrossingDf()
          if "uniqueTrackId" in crossingDf:
            uniqueIds = crossingDf.uniqueTrackId.unique()
            logging.info(f"crossing ids for {recordingData.recordingId}: {recordingData.getCrossingPedIds()}")
            logging.info(f"uniqueIds for {recordingData.recordingId}: {uniqueIds}")
            self.__crossingIds.update(uniqueIds)
          else:
            logging.warn(f"{recordingData.recordingId} does not have uniqueTrackId")
        except Exception as e:
          logging.warn(f"{recordingData.recordingId} has exception: {e}")

    
    return self.__crossingIds


  def getCrossingDf(self):
    if self.__crossingDf is None:
      dfs = []
      for recordingData in self.recordingDataList:
        try:
          dfs.append(recordingData.getCrossingDf())
        except Exception as e:
          logging.warn(f"{recordingData.recordingId} has exception: {e}")

      self.__crossingDf = pd.concat(dfs, ignore_index=True)
    
    return self.__crossingDf





