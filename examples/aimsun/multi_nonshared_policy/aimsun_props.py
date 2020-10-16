import csv
import sys
sys.path.append('/home/damian/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np
import pandas as pd
import json

class Aimsun_Params:
    
    def __init__(self,my_csv):
        self.df = pd.read_csv(my_csv)

    def get_green_phases(self, node_id):
        gp = self.df.loc[self.df.node_id == node_id, 'green_phases_list'].values[0]
        green_phases = json.loads(gp)
        #gp_list = green_phases[0]
        return green_phases

    def get_cp_cycle_dict(self, node_id, rep_name):
        cp_cycle_dict = {}
        df_cp = self.df[self.df['node_id'] == node_id]
        cp = df_cp.loc[df_cp.rep_name == rep_name, 'cp_id_list'].values[0]
        cp = json.loads(cp)

        df_cycle = self.df[self.df['node_id'] == node_id]
        cycle = df_cycle.loc[df_cycle.rep_name == rep_name, 'cycle_list'].values[0]
        cycle = json.loads(cycle)

        cp_cycle_dict = dict(zip(cp, cycle))

        return cp_cycle_dict

    def get_sum_interphase_per_ring(self, node_id):
        s_int = self.df.loc[self.df.node_id == node_id, 'sum_interphase'].values[0]
        sum_interphases = json.loads(s_int)
        return sum_interphases

    def get_max_dict(self, node_id, rep_name):
        in_df = self.df.set_index('node_id')
        b_df = in_df.loc[[node_id],['rep_name','maxd_1','maxd_2','maxd_3','maxd_p']]
        in_df = b_df.set_index('rep_name')
        c_df = in_df.loc[[rep_name],['maxd_1','maxd_2','maxd_3','maxd_p']]
        maxd_1 = [json.loads(c_df.iloc[0]['maxd_1'])]
        maxd_2 = [json.loads(c_df.iloc[0]['maxd_2']) ]
        maxd_3 = [json.loads(c_df.iloc[0]['maxd_3'])]
        maxd_p = json.loads(c_df.iloc[0]['maxd_p'])
        max_dict = sum([maxd_1,maxd_2,maxd_3],[])
        # max_dict = dict(zip(maxd_p, maxd_list))
        return max_dict, maxd_p

    def get_detector_ids(self, node_id):
        pass

    def get_start_phases(self, node_id):
        sp = self.df.loc[self.df.node_id == node_id, 'start_phase'].values[0]
        start_phases = json.loads(sp)
        #gp_list = green_phases[0]
        return start_phases

class Export_Params:
    def __init__(self, rep_name, node_id):
        self.rep_name = str(rep_name) + '_' + str(node_id) + '.csv'
        self.fieldnames = ['time', 'delay_time','action']
        with open(self.rep_name,'a') as csvFile:
            csv_writer = csv.writer(csvFile)
            csv_writer.writerow([node_id])

    def export_delay_action(self, node_id, delay, action_list, util_list, time, timeSta):
        time = time
        timeSta = timeSta
        ave_app_delay = delay
        data_list = [time,delay]
        #data_list.append(time)
        #data_list.append(delay)
        
        for action in action_list:
            data_list.append(action)

        for util in util_list:
            data_list.append(util)

        with open(self.rep_name, 'a') as csvFile:
            csv_writer = csv.writer(csvFile)
            csv_writer.writerows([data_list,])

##test
#print(get_green_phases(3344))
#print('********************')
#ap = Aimsun_Params('/home/cjrsantos/flow/examples/aimsun/single_light/aimsun_props.csv')
#print(ap.get_cp_cycle_dict(3369,8050315), print(type(ap.get_cp_cycle_dict(3369,8050315))))
#print('********************')
#print(get_sum_interphase_per_ring(3344), print(type(get_sum_interphase_per_ring(3344))))
#print('********************')
#max_d, max_p = ap.get_max_dict(3344,8050322)
