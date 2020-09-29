### Export relevant data to csv ###

import AAPI as aapi
import sys
import csv
import PyANGKernel as gk
import PyANGConsole as cs
from datetime import datetime
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np
import pandas as pd 

model = gk.GKSystem.getSystem().getActiveModel()
# global edge_detector_dict
# edge_detector_dict = {}


now = datetime.now()
westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]
sections = [22208, 568, 22211, 400]
target_nodes = [3329, 3344, 3370, 3341, 3369]

interval = 15*60
#seed = np.random.randint(2e9)

replication_name = aapi.ANGConnGetReplicationId()
replication = model.getCatalog().find(8050315)
current_time = now.strftime('%H-%M:%S')

starting_ids = []
cp_am_list = []
cp_nn_list = []
cp_pm_list = []

## another option is to store to dataframe first then convert to csv

with open('{}.csv'.format('aimsun_params_1'), 'a') as csvFile:
    fieldnames = ['rep_name', 'node_id', 'num_rings', 'green_phases_list', 'cp_id_list',
                  'cycle_list', 'sum_interphase', 'maxd_1', 'maxd_2','maxd_3','maxd_p',
                  'det_left','det_right','det_through','det_adv','det_lanes']  # cycle_plan differs per replication
    csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    csv_writer.writeheader()

def get_control_ids(node_id):
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings


def get_cycle_length(node_id, control_id):  # cj
    # Format is set current control plan
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    return control_cycle


def get_green_phases(node_id, ring_id, timeSta):
    a = 1
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    if ring_id > 0:
        a = num_phases + 1
        num_phases = num_phases*2

    return [phase for phase in range(a, num_phases+1) if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 0]


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
            dur, _, _ = get_duration_phase(3344, phase, timeSta)
            idur = int(dur)
            dur_list.append(idur)
    return dur_list, cycle

def get_incoming_edges(node_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()

    return [edge.getId() for edge in in_edges]

def get_detectors_on_edge(edge_id):
    catalog = model.getCatalog()
    detector_list = {"left": [], "right":[], "through":[],"advanced": []}
    section_inf = aapi.AKIInfNetGetSectionANGInf(edge_id)
    for i in range(aapi.AKIDetGetNumberDetectors()):
        detector = aapi.AKIDetGetPropertiesDetector(i)
        if detector.IdSection == edge_id:
            edge_aimsun = catalog.find(detector.IdSection)
            num_lanes = int(section_inf.nbCentralLanes + section_inf.nbSideLanes)

            if (edge_aimsun.length2D() - detector.FinalPosition) < 6 and detector.IdLastLane == num_lanes: # 3or2 for 3370, 4 for 3369
                kind = "left"
            elif (edge_aimsun.length2D() - detector.FinalPosition) < 6 and detector.IdFirstLane == 1 and num_lanes > 3: #0 for 3370, 1 for 3369
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

def get_starting_phases(node_id, timeSta):
    phases
    cid, num_rings = get_control_ids(node_id)
    for i in range(num_rings):
        phases.append(get_green_phases(node_id, i,timeSta))
    num_phases = len(phases)
    if num_phases > 6:
        start_phase = [1,9]
    elif num_phases > 4:
        start_phase = [1,7]
    else:
        start_phase = [1,5]
    
    return start_phase
def get_detector_dict(node_id, detector_dict):
    pass

def AAPILoad():
    return 0


def AAPIInit():
    
    # set_replication_seed(seed)
    # print(seed)
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    #if time == 0:
    #    for node_id in nodes:
    #        global starting_ids
    #        starting_ids.append(aapi.ECIGetNumberCurrentControl(node_id))
    #        phases = []
    #        green_phases_list = []
    #        maxd_list = []
    #        cp_id_list = []
    #        cp_name_list = []
    #        cycle_list = []
    #        cycle_am = []
    #        cycle_noon = []
    #        cycle_pm = []
#
    #        # get number of green phases and ring
    #        cid, num_rings = get_control_ids(node_id)
    #        for i in range(num_rings):
    #            phases.append(get_green_phases(node_id, i,timeSta))
#
    #        for phase_list in phases:
    #            for phase in phase_list:
    #                green_phases_list.append(phase)  # compile all green phases in a list
#
    #        for phase in green_phases_list:
    #            dur, maxd, mind = get_duration_phase(node_id, phase, timeSta)
    #            maxd_list.append(maxd)
#
    #        ##### control plan params ####
    #        num_cp = aapi.ECIGetNumberofControls(node_id)
    #        cp_id_list = list(range(num_cp))
#
    #        for i in range(num_cp):
    #            cycle_list.append(aapi.ECIGetControlCycleofJunction(i, node_id))
#
    #        with open('{}.csv'.format('aimsun_params'), 'a') as csvFile:
    #            csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    #            csv_writer.writerow({'node_id': node_id, 'num_rings': num_rings, 'green_phases_list': green_phases_list,
    #                                'cp_id_list': cp_id_list, 'cycle_list': cycle_list})
#
    #            # read control plan id
    #            # if control plan changes append id and cycle and max d

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
