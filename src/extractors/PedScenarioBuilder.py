class PedScenarioBuilder:

    """
        for each crossing uniqueTrackId
            for its recordingId, start and end 
                generate a scenario id,
                get all the rows overlapping the crossing frames
                filter out the trajectories that does not overlap with the scene config
                remove own frames
                extract types and ids of other tracks
                build PedScenario ob
    """

    def __init__(
            self
        ):

        self._nextId = 0


    def getNextId(self):
        self._nextId += 1
        return self._nextId
    

    def buildFromSceneData(self, locationData, sceneData):
