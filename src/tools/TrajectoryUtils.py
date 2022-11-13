from shapely.geometry import LineString, box, Point
from shapely.affinity import rotate, translate, affine_transform

from tqdm import tqdm
import pandas as pd
import numpy as np
from math import sqrt, inf, cos, sin, radians
from typing import List
import vg

class TrajectoryUtils:

    @staticmethod
    def length(trajectoryDf: pd.DataFrame, xCol, yCol) -> int:
        spline = TrajectoryUtils.dfToSplines(trajectoryDf, xCol, yCol)
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

            distance = sqrt((last[0] - prev[0]) ** 2 +
                            (last[1] - prev[1]) ** 2)
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
    def getDfIfDfIntersect(sceneId, sceneConfig, scenePolygon: box, df: pd.DataFrame, xCol="xCenter", yCol="yCenter") -> bool:
        trajSpline = TrajectoryUtils.dfToSplines(df, xCol, yCol, 1)
        if TrajectoryUtils.doesIntersect(scenePolygon, trajSpline):
            df = df.copy()  # we can modify without concern now
            df["sceneId"] = sceneId
            df["roadWidth"] = sceneConfig["roadWidth"]
            return df
        return None
    

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

            if entryFrame > 0 and exitFrame == inf:
                # check if this row is an exit point
                if not rect.contains(Point(row[xCol], row[yCol])):
                    exitFrame = row[frameCol]
                    break

        # sometimes there are no exit frame. use the last frame
        if exitFrame == inf:
            exitFrame = row[frameCol]

        return pedDf[(pedDf[frameCol] >= entryFrame) & (pedDf[frameCol] <= exitFrame)]

    @staticmethod
    def clipByRect(pedDf, xCol, yCol, frameCol, rect) -> pd.DataFrame:
        """ Clip the trajectory with 150% rect clipping. """

        # find entry and exit point frame number, keep all the points in between and disregard others. A trajectory may enter several times, but we don't need them.

        entryFrame = -inf
        exitFrame = inf

        for idx, row in pedDf.iterrows():
            if entryFrame == -inf:
                # check if this row is an entry point
                if rect.contains(Point(row[xCol], row[yCol])):
                    entryFrame = row[frameCol]
                    continue

            if entryFrame > 0 and exitFrame == inf:
                # check if this row is an exit point
                if not rect.contains(Point(row[xCol], row[yCol])):
                    exitFrame = row[frameCol]
                    break

        # sometimes there are no exit frame. use the last frame
        if exitFrame == inf:
            exitFrame = row[frameCol]

        return pedDf[(pedDf[frameCol] >= entryFrame) & (pedDf[frameCol] <= exitFrame)]

    @staticmethod
    def getTranslationMatrix(localCenterPosition: Point) -> List[float]:

        return [1, 0, 0, 1, -localCenterPosition.x, -localCenterPosition.y]

    @staticmethod
    def getRotationMatrix(localCenterRotation) -> List[float]:
        """
        returns rotationMatrix wrt 0,0, not local coordinate system.
        """
        rotInRad = -radians(localCenterRotation)
        costTheta = cos(rotInRad)
        sinTheta = sin(rotInRad)
        return [costTheta, -sinTheta, sinTheta, costTheta, 0, 0]

    @staticmethod
    def transformPoint(translationMatrix, rotationMatrix, point: Point) -> Point:

        translated = affine_transform(point, translationMatrix)
        return affine_transform(translated, rotationMatrix)  # order matters.

    
    @staticmethod
    def getType(trajDf: pd.DataFrame) -> str:
        head = trajDf.head(1).to_dict("records")[0]
        return head["class"]


    # region trajectory interactions

    @staticmethod
    def doPathsIntersect(traj1: pd.DataFrame, traj2: pd.DataFrame, xCol="xCenter", yCol="yCenter"):
        """
            Useful to check if a vehicle is oncoming
        """
        spline1 = TrajectoryUtils.dfToSplines(traj1, xCol, yCol, 1)
        spline2 = TrajectoryUtils.dfToSplines(traj2, xCol, yCol, 1)
        return spline1.intersects(spline2)

    @staticmethod
    def minPathDistance(traj1: pd.DataFrame, traj2: pd.DataFrame, xCol="xCenter", yCol="yCenter") -> float:
        """This method is naive, we should get the minimum common frame, and use kalman filter to predict future path using a few frames

        Args:
            traj1 (pd.DataFrame): _description_
            traj2 (pd.DataFrame): _description_
            xCol (str, optional): _description_. Defaults to "xCenter".
            yCol (str, optional): _description_. Defaults to "yCenter".
        """
        spline1 = TrajectoryUtils.dfToSplines(traj1, xCol, yCol, 1)
        spline2 = TrajectoryUtils.dfToSplines(traj2, xCol, yCol, 1)
        return spline1.distance(spline2)

    @staticmethod
    def sameDirection(traj1: pd.DataFrame, traj2: pd.DataFrame, xCol="xCenter", yCol="yCenter"):
        """compares the first spline

        Args:
            traj1 (pd.DataFrame): _description_
            traj2 (pd.DataFrame): _description_
            xCol (str, optional): _description_. Defaults to "xCenter".
            yCol (str, optional): _description_. Defaults to "yCenter".
        """
        spline1 = TrajectoryUtils.dfToSplines(traj1, xCol, yCol, 1)
        spline2 = TrajectoryUtils.dfToSplines(traj2, xCol, yCol, 1)
        raise NotImplementedError("sameDirection")

    @staticmethod
    def minSplineDistance(spline1: LineString, spline2: LineString) -> float:
        return spline1.distance(spline2)
    
    @staticmethod
    def splineToFirstVector(spline: LineString) -> np.array:
        return np.asarray((spline.coords[1][0] - spline.coords[0][0], spline.coords[1][1] - spline.coords[1][1]))

    @staticmethod
    def sameDirectionSplines(spline1: LineString, spline2: LineString):

        """Naive approach

        Returns:
            _type_: _description_
        """
        
        v0 = TrajectoryUtils.splineToFirstVector(spline1)
        v1 = TrajectoryUtils.splineToFirstVector(spline2)
        # angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        angle = vg.angle(v0, v1, units="deg")
        if angle < 90:
            return False
        return True

    #endregion

    @staticmethod
    def downSample(traj: pd.DataFrame, fromFPS:float, toFPS: float):
        if fromFPS < toFPS:
            raise Exception(f"downSample: Up sampling not supported for frames")
        if fromFPS == toFPS:
            return traj
        
        keepInterval = fromFPS // toFPS

        downTraj = []
        count = 0
        for _, row in  tqdm(traj.iterrows(), desc=f"downsampling", total=len(traj)):
            if count % keepInterval == 0:
            # if row["trackLifetime"] % keepInterval == 0: # a little error will break it
                downTraj.append(row)
            count += 1
        
        return pd.DataFrame(downTraj).convert_dtypes()

    @staticmethod
    def downSampleByTrackLifeTime(traj: pd.DataFrame, fromFPS:float, toFPS: float):
        if fromFPS < toFPS:
            raise Exception(f"downSample: Up sampling not supported for frames")
        if fromFPS == toFPS:
            return traj
        
        keepInterval = fromFPS // toFPS

        downTraj = []
        for _, row in  tqdm(traj.iterrows(), desc=f"downsampling", total=len(traj)):
        # for _, row in traj.iterrows():
            if row["trackLifetime"] % keepInterval == 0: # a little error will break it
                downTraj.append(row)
        
        return pd.DataFrame(downTraj).convert_dtypes()


        