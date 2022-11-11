class PedScenarioBuilder:

    """
        for each recording
            for each crossing ped
                generate a scenario id,
                get all the rows overlapping the crossing frames
                remove own frames
                extract types and ids of other tracks
                build PedScenario ob
    """