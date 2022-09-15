from shapely.geometry import LineString, box, Point
from shapely.affinity import rotate

import pandas as pd
from math import sqrt, inf

class TrajectoryUtils:

  @staticmethod
  def length(trajectoryDf: pd.DataFrame, xCol, yCol) -> int:
    spline =  TrajectoryUtils.dfToSplines(trajectoryDf, xCol, yCol)
    return spline.length

  @staticmethod
  def dfToSplines(trajectoryDf: pd.DataFrame, xCol, yCol, minLen=0.5) -> LineString:
    """
    Arguments:
      minLen: minimum spline size
    """
    splinePoints = []
    prev = None
    last = None
    for idx, row in trajectoryDf.iterrows():
      last = (row[xCol], row[yCol])
      if prev is None:
        prev = last
        continue

      distance = sqrt((last[0] - prev[0]) ** 2 + (last[1] - prev[1]) ** 2) 
      if distance >= minLen:
        splinePoints.append(prev)
        prev = last


    # add the last points
    splinePoints.append(prev)
    splinePoints.append(last)

    return LineString(splinePoints)

  @staticmethod
  def scenePolygon(sceneConfig, boxWidth, boxHeight) -> box:
    # try with a rect box, minx, miny, maxx, maxy
    minX = sceneConfig['centerX'] - boxWidth / 2
    minY = sceneConfig['centerY'] - boxHeight / 2
    maxX = sceneConfig['centerX'] + boxWidth / 2
    maxY = sceneConfig['centerY'] + boxHeight / 2

    # TODO rotate

    bbox = box(minX, minY, maxX, maxY)
    return rotate(bbox, sceneConfig['angle'])

  
  @staticmethod
  def doesIntersect(polygon: box, spline: LineString) -> bool:
    return polygon.intersects(spline)

  
  @staticmethod
  def clip(pedDf, xCol, yCol, frameCol, sceneConfig, boxWidth, boxHeight) -> pd.DataFrame:
    """ Clip the trajectory with 150% rect clipping. """

    rect = TrajectoryUtils.scenePolygon(sceneConfig, boxWidth, boxHeight)
    # find entry and exit point frame number, keep all the points in between and disregard others. A trajectory may enter several times, but we don't need them.

    entryFrame = -inf
    exitFrame = inf

    for idx, row in pedDf.iterrows():
      if entryFrame == -inf:
        # check if this row is an entry point
        if rect.contains(Point(row[xCol], row[yCol])):
          entryFrame = row[frameCol]
          continue

      if entryFrame > 0  and exitFrame == inf:
        # check if this row is an exit point
        if not rect.contains(Point(row[xCol], row[yCol])):
          exitFrame = row[frameCol]
          break
    
    # sometimes there are no exit frame. use the last frame
    if exitFrame == inf:
      exitFrame = row[frameCol]

    return pedDf[(pedDf[frameCol] >= entryFrame) & (pedDf[frameCol] <= exitFrame)]

