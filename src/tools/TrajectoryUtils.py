from shapely.geometry import LineString, box, Point
from shapely.affinity import rotate, translate, affine_transform
from extractors.TrackDirection import TrackDirection

from tqdm import tqdm
import pandas as pd
import numpy as np
from math import sqrt, inf, cos, sin, radians
from typing import List, Tuple
import vg
import logging


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
    def clipByRect(pedDf, xCol, yCol, frameCol, rect) -> Tuple[pd.DataFrame, int]:
        """ Clip the trajectory with 150% rect clipping. Returns how many times the trajectory exitted the scene """

        # find entry and exit point frame number, keep all the points in between and disregard others. A trajectory may enter several times, but we don't need them.

        # assert len(pedDf) > 1

        entryFrame = -inf
        exitFrame = inf

        exitCount = 0

        for idx, row in pedDf.iterrows():

            insideRect = rect.contains(Point(row[xCol], row[yCol]))
            if entryFrame == -inf:
                # check if this row is an entry point
                if insideRect:
                    entryFrame = row[frameCol]
                    continue

            if entryFrame > 0:
                if not insideRect:
                    if exitFrame == inf:
                        # check if this row is an exit point
                        # Naive method as the ped can enter again
                        exitFrame = row[frameCol]
                        exitCount += 1
                        # break
                else: # we keep looking for more
                    exitFrame = inf #entered again, so we keep looking
        
        if entryFrame == -inf:
            logging.warn(f"{pedDf.iloc[0]['uniqueTrackId']} has no entry frame in {rect}")
            return None, 0

        # sometimes there are no exit frame. use the last frame
        if exitFrame == inf:
            exitFrame = row[frameCol]
            exitCount += 1
        
        return pedDf[(pedDf[frameCol] >= entryFrame) & (pedDf[frameCol] <= exitFrame)], exitCount

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

    # endregion

    @staticmethod
    def downSample(traj: pd.DataFrame, fromFPS: float, toFPS: float):
        if fromFPS < toFPS:
            raise Exception(
                f"downSample: Up sampling not supported for frames")
        if fromFPS == toFPS:
            return traj

        keepInterval = fromFPS // toFPS

        downTraj = []
        count = 0
        for _, row in tqdm(traj.iterrows(), desc=f"downsampling", total=len(traj)):
            if count % keepInterval == 0:
                # if row["trackLifetime"] % keepInterval == 0: # a little error will break it
                downTraj.append(row)
            count += 1

        return pd.DataFrame(downTraj).convert_dtypes()

    @staticmethod
    def downSampleByTrackLifeTime(traj: pd.DataFrame, fromFPS: float, toFPS: float):
        if fromFPS < toFPS:
            raise Exception(
                f"downSample: Up sampling not supported for frames")
        if fromFPS == toFPS:
            return traj

        keepInterval = fromFPS // toFPS

        downTraj = []
        for _, row in tqdm(traj.iterrows(), desc=f"downsampling", total=len(traj)):
            # for _, row in traj.iterrows():
            if row["trackLifetime"] % keepInterval == 0:  # a little error will break it
                downTraj.append(row)

        return pd.DataFrame(downTraj).convert_dtypes()

    @staticmethod
    def getTrack_VH_Directions(trackDf: pd.DataFrame, xCol, yCol) -> Tuple[TrackDirection, TrackDirection]:
        """_summary_

        Args:
            trackDf (pd.DataFrame): NORTH is positive y, EAST is positive x

        Returns:
            Tuple[TrackDirection]: NORTH/SOUTH, EAST/WEST
        """
        # if local y is decreasing, then SOUTH
        # if local x is increasing, then EAST
        verticalDirection = TrackDirection.NORTH
        horizontalDirection = TrackDirection.EAST
        firstRow = trackDf.head(1).iloc[0]
        lastRow = trackDf.tail(1).iloc[0]
        if firstRow[yCol] > lastRow[yCol]:
            verticalDirection = TrackDirection.SOUTH

        if firstRow[xCol] > lastRow[xCol]:
            horizontalDirection = TrackDirection.WEST

        return verticalDirection, horizontalDirection

    @staticmethod
    def getTimeDerivativeForOne(aTrack: pd.DataFrame, onCol, fps):
        derivativeSeries = aTrack[onCol].rolling(window=2).apply(
            lambda values: (values.iloc[0] - values.iloc[1]) / (1 / fps))
        derivativeSeries.iloc[0] = derivativeSeries.iloc[1]
        return derivativeSeries

    @staticmethod
    def getVelocitySeriesForOne(aTrack: pd.DataFrame, onCol, fps):
        return TrajectoryUtils.getTimeDerivativeForOne(aTrack, onCol, fps)
        # seriesVelo = aTrack[onCol].rolling(window=2).apply(
        #     lambda values: (values.iloc[0] - values.iloc[1]) / (1 / fps))
        # seriesVelo.iloc[0] = seriesVelo.iloc[1]
        # return seriesVelo

    @staticmethod
    def getVelocitySeriesForAll(tracksDf: pd.DataFrame, onCol, fps):
        individualSeres = []
        # for trackId in tqdm(tracksDf["uniqueTrackId"].unique(), desc=f"deriving velocity on {onCol} at fps {fps}", position=0):
        for trackId in tracksDf["uniqueTrackId"].unique():
            aTrack = tracksDf[tracksDf["uniqueTrackId"] == trackId]
            individualSeres.append(
                TrajectoryUtils.getTimeDerivativeForOne(aTrack, onCol, fps))

        velSeries = pd.concat(individualSeres)
        return velSeries

    @staticmethod
    def getAccelerationSeriesForAll(tracksDf: pd.DataFrame, onCol, fps):
        individualSeres = []
        # for trackId in tqdm(tracksDf["uniqueTrackId"].unique(), desc=f"deriving acceleration on {onCol} at fps {fps}", position=0):
        for trackId in tracksDf["uniqueTrackId"].unique():
            aTrack = tracksDf[tracksDf["uniqueTrackId"] == trackId]
            individualSeres.append(
                TrajectoryUtils.getTimeDerivativeForOne(aTrack, onCol, fps))

        accSeries = pd.concat(individualSeres)
        return accSeries

    @staticmethod
    def getAccelerationSerieFromVelocityForOne(velocitySeries: pd.Series, fps):
        seriesAcc = velocitySeries.rolling(window=2).apply(
            lambda values: (values.iloc[0] - values.iloc[1]) / (1 / fps))
        seriesAcc.iloc[0] = seriesAcc.iloc[1]
        return seriesAcc

        pass

    @staticmethod
    def trimHeadAndTailForAll(tracksDf: pd.DataFrame):
        trimmedTracks = []
        # for trackId in tqdm(tracksDf["uniqueTrackId"].unique(), desc=f"trimming trajectories"):
        for trackId in tracksDf["uniqueTrackId"].unique():
            aTrack = tracksDf[tracksDf["uniqueTrackId"] == trackId]
            trimmedTracks.append(aTrack.iloc[2: len(aTrack) - 2, :]) # 4 frames to exclude invalid acceleration and velocities
        
        return pd.concat(trimmedTracks)

    @staticmethod
    def rotate(trackDf: pd.DataFrame, origin: Point = None):
        if origin is not None:
            raise Exception("trajectory rotate does not support origin")
        pass

    @staticmethod
    def flipY(trackDf: pd.DataFrame, origin: Point = None):
        if origin is not None:
            raise Exception("trajectory flipY does not support origin")
        pass

    @staticmethod
    def flipX(trackDf: pd.DataFrame, origin: Point = None):
        if origin is not None:
            raise Exception("trajectory flipX does not support origin")
        pass
