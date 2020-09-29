import AAPI as aapi
import sys
import csv
import PyANGKernel as gk
import PyANGConsole as cs
from datetime import datetime
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np

model = gk.GKSystem.getSystem().getActiveModel()
# global edge_detector_dict
# edge_detector_dict = {}


now = datetime.now()
westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]
sections = [22208, 568, 22211, 400]
green_phases = [1,3,5,7,9,11]
node_id = 3370

interval = 15*60
#seed = np.random.randint(2e9)

replication_name = aapi.ANGConnGetReplicationId()
replication = model.getCatalog().find(8050315)
current_time = now.strftime('%H-%M:%S')
rep_name = str(replication_name) + '_nopolicy_' + str(node_id) + '.csv'



def get_delay_time(section_id):
    west = []
    east = []
    for section_id in westbound_section:
        estad_w = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
        if estad_w.report == 0:
            print('Delay time: {} - {}'.format(section_id, estad_w.DTa))
        west.append(estad_w.DTa)

    for section_id in eastbound_section:
        estad_e = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
        if estad_e.report == 0:
            print('Delay time: {} - {}'.format(section_id, estad_e.DTa))
        east.append(estad_e.DTa)

    west_ave = sum(west)/len(west)
    east_ave = sum(east)/len(east)

    print("Average Delay Time: WestBound {}".format(west_ave))
    print("Average Delay Time: EastBound {}".format(east_ave))


def sum_queue(section_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()

    section_list = [edge.getId() for edge in in_edges]

    for section_id in section_list:
        section = catalog.find(section_id)
        num_lanes = section.getNbLanesAtPos(section.length2D())
        queue = sum(aapi.AKIEstGetCurrentStatisticsSectionLane(
            section_id, i, 0).LongQueueAvg for i in range(num_lanes))

        queue = queue * 5 / section.length2D()

    print('SUM QUEUE {} : {}'.format(node_id))

def set_replication_seed(seed):
    replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
    for replication in replications.values():
        replication.setRandomSeed(seed)

def get_ttadta(section_id, timeSta):
    # print( "AAPIPostManage" )
    if time % (15*60) == 0:
        for section_id in sections:
            estad = aapi.AKIEstGetParcialStatisticsSection(section_id, timeSta, 0)
            if (estad.report == 0):
                dta = estad.DTa
                tta = estad.TTa
                time = time
                # print('dt: {:.4f}, tt: {:.4f}'.format(estad.DTa, estad.TTa))
                # print('\n Mean Queue: \t {}'.format(estad.))


def get_control_ids(node_id):
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings


def get_cycle_length(node_id, control_id):  # cj
    # Format is set current control plan
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    return control_cycle


def get_phases(ring_id):
    phases = []
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in range(1, num_phases*2+1):
        phases.append(phase)
    return phases

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

def get_phase_duration_list(node_id, timeSta):
    control_id, num_rings = get_control_ids(node_id)
    cycle = get_cycle_length(node_id, control_id)
    phase_list = get_phases(0)
    dur_list = []
    for phase in phase_list:
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
            continue
        else:
            dur, _, _ = get_duration_phase(node_id, phase, timeSta)
            idur = int(dur)
            dur_list.append(idur)
    return dur_list, cycle

def get_replication_name(node_id): #cj28
    node_id = node_id
    rep_name = aapi.ANGConnGetReplicationId()

    replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
    for replication in replications.values():
        rep_seed = replication.getRandomSeed()

    return rep_name, rep_seed

def export_delay_action(node_id, delay, action_list, time, timeSta):
    time = time
    timeSta = timeSta
    ave_app_delay = delay
    data_list = [time,delay]
    
    for action in action_list:
        data_list.append(action)

    with open(rep_name, 'a') as csvFile:
        csv_writer = csv.writer(csvFile)
        csv_writer.writerows([data_list,])

def AAPILoad():
    return 0


def AAPIInit():
    #set_replication_seed(seed)
    name,seed = get_replication_name(node_id)
    print(name,seed)
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIManage" )
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIPostManage" )
    if time % 900 == 0:
        action_list = []
        for phase in green_phases:
            normalDuration, _, _ = get_duration_phase(node_id, phase, timeSta)
            action_list.append(normalDuration)
        delay = aapi.AKIEstGetPartialStatisticsNodeApproachDelay(node_id)
        export_delay_action(node_id, delay, action_list, time, timeSta)

    """# console = cs.ANGConsole()
    if time == interval:
        print('yey')
        aapi.ANGSetSimulationOrder(1, interval)
        # aapi.ANGSetSimulationOrder(2, 0)
        # replication = model.getCatalog().find(replication_name)
        gk.GKSystem.getSystem().executeAction("execute", replication, [], "")
        # console.close()"""
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
