import AAPI as aapi


def change_phase_duration(node_id, phase, duration, time, timeSta, acycle):
    aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)


def get_duration_phase(node_id, phase, timeSta):
    normalDurationP = aapi.doublep()
    maxDurationP = aapi.doublep()
    minDurationP = aapi.doublep()
    aapi.ECIGetDurationsPhase(node_id, phase, timeSta,
                              normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()

    return normalDuration, maxDuration, minDuration


def AAPILoad():
    """Execute commands while the Aimsun template is loading."""
    return 0


def AAPIInit():
    """Execute commands while the Aimsun instance is initializing."""
    # set the simulation time to be very large
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    """Execute commands before an Aimsun simulation step."""
    # Create a thread when data needs to be sent back to FLOW
    if time % 900 == 0:
        phase_list = [1, 3, 5, 7, 9, 11]

        duration = [10, 10, 20, 10, 10, 20]

        for phase, dur in zip(phase_list, duration):
            change_phase_duration(3341, phase, dur, time, timeSta, acycle)
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    """Execute commands after an Aimsun simulation step."""
    if time % 900 == 0:
            nd, _, _ = get_duration_phase(3341, 11, timeSta)
            print(nd)
    return 0


def AAPIFinish():
    """Execute commands while the Aimsun instance is terminating."""
    return 0


def AAPIUnLoad():
    """Execute commands while Aimsun is closing."""
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    """Execute Aimsun route choice calculation."""
    return 0


def AAPIEnterVehicle(idveh, idsection):
    """Execute command once a vehicle enters the Aimsun instance."""
    return 0


def AAPIExitVehicle(idveh, idsection):
    """Execute command once a vehicle exits the Aimsun instance."""
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    """Execute command once a pedestrian enters the Aimsun instance."""
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    """Execute command once a pedestrian exits the Aimsun instance."""
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    """Execute command once a vehicle enters the Aimsun instance."""
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    """Execute command once a vehicle exits the Aimsun instance."""
    return 0
