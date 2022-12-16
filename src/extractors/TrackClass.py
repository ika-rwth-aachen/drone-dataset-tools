from enum import Enum
import pandas as pd

class TrackClass(Enum):
    Car = 'car'
    Bicycle = 'bicycle'
    Pedestrian = 'pedestrian'
    Truck_Bus = 'truck_bus'
    FastPedestrian = 'fast_pedestrian'

    @staticmethod
    def getTrackType(otherDf: pd.DataFrame):
        otherClasses = otherDf["class"].unique()
        assert len(otherClasses) == 1
        return otherClasses[0]

    @staticmethod
    def isPedestrian(otherDf: pd.DataFrame):
        return TrackClass.getTrackType(otherDf) == TrackClass.Pedestrian.value

    @staticmethod
    def isFastPedestrian(otherDf: pd.DataFrame):
        return TrackClass.getTrackType(otherDf) == TrackClass.FastPedestrian.value

    @staticmethod
    def isBicycle(otherDf: pd.DataFrame):
        return TrackClass.getTrackType(otherDf) == TrackClass.Bicycle.value

    @staticmethod
    def isCar(otherDf: pd.DataFrame):
        return TrackClass.getTrackType(otherDf) == TrackClass.Car.value

    @staticmethod
    def isLargeVehicle(otherDf: pd.DataFrame):
        return TrackClass.getTrackType(otherDf) == TrackClass.Truck_Bus.value