from shapely.geometry import LineString, box
import pandas as pd
from math import sqrt

class TrajectoryUtils:

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

    return box(minX, minY, maxX, maxY)

  
  @staticmethod
  def doesIntersect(polygon: box, spline: LineString) -> bool:
    return polygon.intersects(spline)


