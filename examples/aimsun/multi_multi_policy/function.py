import AAPI as aapi
import PyANGKernel as gk

model = gk.GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}
length_car = 5  # typical car length


def get_incoming_edges(node_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()

    return [edge.getId() for edge in in_edges]


def get_detector_ids(edge_id):
    catalog = model.getCatalog()
    detector_list = {"left": [], "right": [], "through": [], "advanced": []}
    for i in range(aapi.AKIDetGetNumberDetectors()):
        detector = aapi.AKIDetGetPropertiesDetector(i)
        if detector.IdSection == edge_id:
            edge_aimsun = catalog.find(detector.IdSection)
            print(detector.Id, detector.IdFirstLane)

            if (edge_aimsun.length2D() - detector.FinalPosition) < 6 and (detector.IdFirstLane == 3 or detector.IdFirstLane == 2):
                kind = "left"
            elif (edge_aimsun.length2D() - detector.FinalPosition) < 6 and detector.IdFirstLane == 0:
                kind = "right"
            elif (edge_aimsun.length2D() - detector.FinalPosition) < 6:
                kind = "through"
            else:
                kind = "advanced"

            detector_obj = catalog.find(detector.Id)
            try:
                # only those with numerical exernalIds are real
                int(detector_obj.getExternalId())
                detector_list[kind].append(detector.Id)
            except ValueError:
                pass
    return detector_list


def AAPILoad():
    return 0


def AAPIInit():
    # print(get_detector_ids(24660))
    # print(get_detector_ids(655))
    print(get_detector_ids(461))
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIPostManage" )
    return 0


def AAPIFinish():
    # print("AAPIFinish")
    return 0


def AAPIUnLoad():
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0
