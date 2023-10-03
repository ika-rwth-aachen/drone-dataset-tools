import sys
from loguru import logger
from pathlib import Path
import json


class UnitUtils:

    @staticmethod
    def InDPixelToCoordinate(pixel, orthoPxToMeter):
        multiplier = orthoPxToMeter * 12
        return (pixel[0] * multiplier, -pixel[1] * multiplier)

    @staticmethod
    def InDCoordinateToPixel(coordinate, orthoPxToMeter):
        multiplier = orthoPxToMeter * 12
        return (coordinate[0] / multiplier, -coordinate[1] / multiplier)

    @staticmethod
    def loadDataSetParams(dataset="ind"):
        # Load dataset specific visualization parameters from file
        dataset_params_path = Path(
            "../data/visualizer_params") / "visualizer_params.json"

        if not dataset_params_path.exists():
            logger.error(
                "Could not find dataset visualization parameters in {}", dataset_params_path)
            raise Exception(
                "Could not find dataset visualization parameters in {}", dataset_params_path)
            # sys.exit(-1)

        with open(dataset_params_path) as f:
            dataset_params = json.load(f)

        if dataset not in dataset_params["datasets"]:
            logger.error("Visualization parameters for dataset {} not found in {}. Please make sure, that the needed "
                         "parameters are given", dataset, dataset_params_path)
            raise Exception(
                "Could not find dataset visualization parameters in {}", dataset_params_path)
            # sys.exit(-1)

        return dataset_params["datasets"][dataset]

    @staticmethod
    def loadSceneConfiguration(dataset="ind"):
        # Load dataset specific visualization parameters from file
        scenePath = Path("../data/scenes") / f"{dataset}.json"

        if not scenePath.exists():
            logger.error("scenePath does not exist - {}", scenePath)
            raise Exception("scenePath does not exist - {}", scenePath)
            # sys.exit(-1)

        with open(scenePath) as f:
            sceneConfig = json.load(f)

        # if dataset not in dataset_params["datasets"]:
        #     logger.error("Scene config for dataset {} not found in {}. Please make sure, that the needed "
        #                   "parameters are given", sceneConfig, scenePath)
        #     sys.exit(-1)

        return sceneConfig

    @staticmethod
    def getLocationSceneConfigs(dataset="ind", locationId=2):
        allLocationSceneConfig = UnitUtils.loadSceneConfiguration(dataset)
        return allLocationSceneConfig[str(locationId)]