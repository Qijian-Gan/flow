import csv
import AAPI as aimsun_api
import PyANGKernel as gk
from collections import OrderedDict
import random as r
import sys
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np

from aimsun_props import Aimsun_Params, Export_Params

ap = Aimsun_Params('/home/cjrsantos/flow/flow/utils/aimsun/aimsun_props.csv')
## Export files
writeFlag = False

model = gk.GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}

target_nodes = [3329, 3344, 3370, 3341, 3369]
start_time = [0]*2
ut_time = [0]*2
green_phases = dict.fromkeys(target_nodes)
starting_phases = dict.fromkeys(target_nodes)
time_consumed = dict.fromkeys(target_nodes,0)
occurence = dict.fromkeys(target_nodes,0)
phaseUtil = dict.fromkeys(target_nodes,0)

for node_id in target_nodes:
    green_phase_list = ap.get_green_phases(node_id)
    starting_phases_list = ap.get_start_phases(node_id)
    starting_phases[node_id] = starting_phases_list
    green_phases[node_id] = green_phase_list
    time_consumed[node_id] = dict.fromkeys(green_phase_list,0)
    occurence[node_id] = dict.fromkeys(green_phase_list,0) # dictionary of node and their phases {node_id:None,...}

if writeFlag == True:
    rep_name = aimsun_api.ANGConnGetReplicationId()
    export_params = Export_Params(rep_name,node_id)

def get_duration_phase(node_id, phase, timeSta):
    normalDurationP = aimsun_api.doublep()
    maxDurationP = aimsun_api.doublep()
    minDurationP = aimsun_api.doublep()
    aimsun_api.ECIGetDurationsPhase(node_id, phase, timeSta,
                              normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()

    return normalDuration, maxDuration, minDuration

def gUtil_at_interval(node_id, ttime, occurs, timeSta):
    global phaseUtil
    action_duration = []
    phase_util = []
    delta = 1e-3
    phase_list = green_phases[node_id]
    print(phase_list)
    for phase in phase_list:
        normalDuration, _, _ = get_duration_phase(node_id, phase, timeSta)
        action_duration.append(normalDuration)

    generated_Duration = action_duration
    control_id = aimsun_api.ECIGetNumberCurrentControl(node_id)
    # what i need is the time_consumed, occurence, generated_duration
    gp_ttime = list(ttime[node_id].values()) #list of total times
    gp_occur = list(occurs[node_id].values()) # list of no. occurences
    gen_dur = generated_Duration # list of generated action for the interval

    for tsecs, occur, dur in zip(gp_ttime, gp_occur, gen_dur):
        try:
            mean_t = tsecs/occur
        except ZeroDivisionError:
            mean_t = 0
        util = (abs(mean_t - dur))/(dur + delta)
        #print(mean_t, dur, util)
        phase_util.append(util)

    node_gutil = sum(phase_util)
    phaseUtil[node_id] = node_gutil
    #print(gp_ttime, gp_occur, gen_dur)
    
    return node_gutil

def get_current_phase(node_id):
    num_rings = aimsun_api.ECIGetCurrentNbRingsJunction(node_id)
    num_phases = [0]*num_rings
    curr_phase = [None]*num_rings
    for ring_id in range(num_rings):
        num_phases[ring_id] = aimsun_api.ECIGetNumberPhasesInRing(node_id, ring_id)
        curr_phase[ring_id] = aimsun_api.ECIGetCurrentPhaseInRing(node_id, ring_id)
        if ring_id > 0:
            curr_phase[ring_id] += num_phases[ring_id]
    return curr_phase

def get_green_time(node_id, time, timeSta):
    #initialize values
    cur_phases = get_current_phase(node_id)
    global ut_time, start_time, time_consumed, occurence, starting_phases
    start_phases = starting_phases[node_id]

    for i, (cur_phase, start_phase) in enumerate(zip(cur_phases, start_phases)):
        if cur_phase != start_phase:
            new_time = round(time)
            ut_time[i] = abs(new_time - start_time[i])
            #print(start_phase,start_time[i], new_time, ut_time[i])
            start_time[i] = new_time
            starting_phases[node_id][i] = cur_phase
            if aimsun_api.ECIIsAnInterPhase(node_id,start_phase,timeSta) == 0:
                time_consumed[node_id][start_phase] += ut_time[i]
                occurence[node_id][start_phase] += 1
                
    return None


def AAPILoad():
    return 0


def AAPIInit():

    return 0



def AAPIManage(time, timeSta, timeTrans, acycle):
    global time_consumed, occurence
    for node_id in target_nodes:
        get_green_time(node_id, time, timeSta)
    if time % 900 == 0:

        gp_Util = []
        for node_id in target_nodes:
            gutil = gUtil_at_interval(node_id, time_consumed, occurence, timeSta)
            gp_Util.append(gutil)
        print(gp_Util)
    

    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    if time % 900 == 0:
        time_consumed = dict.fromkeys(target_nodes,0)
        occurence = dict.fromkeys(target_nodes,0)
        phaseUtil = dict.fromkeys(target_nodes,0)
        #control_id, ring_id, phase_list = get_current_ids(node_id)

    return 0


def AAPIFinish():

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
