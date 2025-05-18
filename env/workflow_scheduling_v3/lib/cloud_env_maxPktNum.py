
import numpy as np
# import pandas as pd
import csv
import math
import os, sys, inspect, random, copy
import gym
import torch

from env.workflow_scheduling_v3.lib.stats import Stats
from env.workflow_scheduling_v3.lib.poissonSampling import one_sample_poisson
from env.workflow_scheduling_v3.lib.vm import VM
from env.workflow_scheduling_v3.lib.workflow import Workflow
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
from env.workflow_scheduling_v3.lib.simsetting import Setting
from env.workflow_scheduling_v3.lib.cal_rank import calPSD


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
vmidRange = 10000


def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_csv_header(file, header):
    ensure_dir_exist(file)
    with open(file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)


def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)


class cloud_simulator(object):
    def __init__(self, args):
        self.set = Setting(args)
        self.baseHEFT = None

        self.trainSet = args["trainSet"]
        self.trainArrivalTime = args["trainArrivalTime"]
        self.testSet = args["testSet"]
        self.testArrivalTime = args["testArrivalTime"]
        self.train_or_test = None

        self.GENindex = None
        self.indEVALindex = None
        self.TaskRule = None

        if self.set.is_wf_trace_record:
            self.df = {}
            # __location__ = os.getcwd() + '\Saved_Results'
            __location__ = os.path.join(os.getcwd(), 'Saved_Results')
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s_gamma%s.csv' % (args["algo"],  args["seed"], args["arrival rate"], args["gamma"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'VM ID', 'VM speed', 'Price', 'VM Rent Start Time', 'VM Rent End Time', 'VM Pending Index'])  # 6 + 6 + 6 columns

        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(6 + self.set.history_len,))
        # self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(9 + self.set.history_len,))
        self.action_space = gym.spaces.Discrete(n=100)

    def close(self):
        print("Environment id %s is closed" % self.set.envid)

    def _init(self):
        self.appSubDeadline = {}
        self.usr_queues = []
        self.vm_queues = []
        self.vm_queues_id = []
        self.vm_queues_cpu = []
        self.vm_queues_rentEndTime = []
        self.usrNum = self.set.usrNum
        self.dcNum = self.set.dcNum
        self.wrfNum = self.set.wrfNum
        self.totWrfNum = self.set.totWrfNum
        self.VMtypeNum = len(self.set.dataset.vmVCPU)
        self.numTimestep = 0
        self.completedWF = 0
        self.VMRemainingTime = {}
        self.VMRemainAvaiTime = {}
        self.VMrentInfos = {}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wrfNum, self.set.history_len)) 
        self.VMcost = 0
        self.SLApenalty = 0
        self.wrfIndex = 0
        self.usrcurrentTime = np.zeros(self.usrNum)
        self.remainWrfNum = 0
        self.missDeadlineNum = 0
        self.VMrentHours = 0  
        self.VMexecHours = 0  

        # IMPORTANT: used to get the ready task for the next time step
        self.firstvmWrfLeaveTime = []
        self.firstusrWrfGenTime = np.zeros(self.usrNum)

        self.uselessAllocation = 0
        self.VMtobeRemove = None

        self.usr_respTime = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wrfNum)) 
        self.usr_sent_pktNum = np.zeros((self.usrNum, self.dcNum))

        for i in range(self.usrNum):
            if self.train_or_test == "train":
                Data_Set = self.trainSet
            elif self.train_or_test == "test":
                Data_Set = self.testSet
            self.usr_queues.append(SimQueue())
            workflowsIDs = Data_Set[self.GENindex][self.indEVALindex]
            for index, appID in np.ndenumerate(workflowsIDs):
                self.workflow_generator(i, appID, index[0])
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstPktEnqueueTime()

        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr()
        self.PrenextTimeStep = self.nextTimeStep
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt()
        temp = self.nextWrf.get_allnextTask(self.finishTask)
                
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]
        if len(temp) > 1:
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True  # decide whether the nextWrf should be dequeued
            self.isNextTaskParallel = False

        self.stat = Stats(self.set)

    # set the arrival time during training and testing for producible
    def workflow_generator(self, usr, appID, index):
        nextArrivalTime = None
        if self.train_or_test == "train":
            nextArrivalTime = self.trainArrivalTime[self.GENindex, self.indEVALindex, index]
        elif self.train_or_test == "test":
            nextArrivalTime = self.testArrivalTime[0, self.indEVALindex, index]

        wrf = self.set.dataset.wset[appID]
        self.remainWrfNum += 1
        pkt = Workflow(self.usrcurrentTime[usr], wrf, appID, usr, self.set.dataset.wsetSlowestT[appID],
                       self.set.dueTimeCoef[usr, appID],
                       self.wrfIndex)
        self.usr_queues[usr].enqueue(pkt, self.usrcurrentTime[usr], None, usr,
                                     0)  # None means that workflow has not started yet
        self.usrcurrentTime[usr] = nextArrivalTime
        self.totWrfNum -= 1
        self.wrfIndex += 1

    def reset(self, seed, ep_num, train_or_test):
        # the seed of env should fix across different training seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.GENindex = seed
        self.indEVALindex = ep_num
        self.train_or_test = train_or_test
        self._init()

    def input_task_rule(self, rule):
        self.TaskRule = rule

    def generate_vmid(self):
        vmid = np.random.randint(vmidRange, size=1)[0]
        while vmid in self.VMRemainingTime:
            vmid = np.random.randint(vmidRange, size=1)[0]
        return vmid

    def get_nextWrfFromUsr(self):
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstPktTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstPktTime

    def get_nextWrfFromVM(self):
        if len(self.firstvmWrfLeaveTime) > 0:
            vmInd = np.argmin(self.firstvmWrfLeaveTime)
            firstPktTime = self.firstvmWrfLeaveTime[vmInd]
            return vmInd, firstPktTime
        else:
            return None, math.inf

    def get_nextTimeStep(self):
        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromVM() 
        if tempnextTimeStep > tempnextTimeStep1:
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.vm_queues[self.nextUsr].get_firstDequeueTask()
        else:
            if tempnextTimeStep == math.inf:
                self.nextTimeStep = None
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep
                self.nextisUsr = True
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstPkt()

    def update_VMRemain_infos(self):
        for key in self.VMRemainingTime:
            ind = self.vm_queues_id.index(key)
            self.VMRemainingTime[key] = self.vm_queues_rentEndTime[ind] - self.nextTimeStep
            maxTimeStep = max(self.vm_queues[ind].currentTimeStep, self.nextTimeStep)
            self.VMRemainAvaiTime[key] = self.vm_queues_rentEndTime[ind] - maxTimeStep - self.vm_queues[ind].vmQueueTime() 

    def remove_expired_VMs(self):
        removed_keys = []        
        Indexes = np.where(np.array(self.vm_queues_rentEndTime) < self.nextTimeStep + 0.00001)[0]

        if not self.nextisUsr:
            nextvmid = self.vm_queues[self.nextUsr].vmid
            if self.nextUsr in Indexes:
                self.VMtobeRemove = nextvmid
            else:
                self.VMtobeRemove = None

        for ind in Indexes:  
            a = self.vm_queues[ind].get_pendingTaskNum()
            if a == 0:
                removed_keys.append(self.vm_queues_id[ind])

        for key in removed_keys:
            del self.VMRemainingTime[key]
            del self.VMRemainAvaiTime[key]
            ind = self.vm_queues_id.index(key)
            del self.vm_queues_id[ind]
            del self.vm_queues_cpu[ind]
            del self.vm_queues_rentEndTime[ind]
            vm = self.vm_queues.pop(ind)
            del self.firstvmWrfLeaveTime[ind]
            del vm              

        if not self.nextisUsr:
            if nextvmid in self.vm_queues_id:
                self.nextUsr = self.vm_queues_id.index(nextvmid)   
            else:
                print('nextvmid is not in self.vm_queues_id')
                self.uselessAllocation += 1
                print('-----> wrong index:', self.uselessAllocation)

    def extend_specific_VM(self, VMindex):
        key = self.vm_queues_id[VMindex]
        maxTimeStep = max(self.vm_queues[VMindex].currentTimeStep, self.nextTimeStep)
        self.VMRemainAvaiTime[key] = self.vm_queues_rentEndTime[VMindex] - maxTimeStep - self.vm_queues[VMindex].vmQueueTime()
        while self.VMRemainAvaiTime[key] < -0.00001:  # ignore system error
            self.VMRemainAvaiTime[key] += self.set.VMpayInterval
            self.vm_queues[VMindex].update_vmRentEndTime(self.set.VMpayInterval)
            self.vm_queues_rentEndTime[VMindex] = self.vm_queues[VMindex].rentEndTime
            self.update_VMcost(self.vm_queues[VMindex].loc, self.vm_queues[VMindex].cpu, True) 
            self.VMrentInfos[key] = self.VMrentInfos[key][:4] + [self.vm_queues[VMindex].rentEndTime]

    def record_a_completed_workflow(self, ddl_penalty):
        if self.set.is_wf_trace_record:        
            Workflow_Infos = [self.nextWrf.appArivalIndex, self.nextWrf.appID,
                              self.nextWrf.generateTime, self.nextTimeStep, self.nextWrf.deadlineTime, ddl_penalty]

            for task in range(len(self.nextWrf.executeTime)):
                Task_Infos = [task, self.nextWrf.app.nodes[task]['processTime'], self.nextWrf.executeTime[task],
                              self.nextWrf.readyTime[task], self.nextWrf.enqueueTime[task], self.nextWrf.dequeueTime[task]]

                VM_Infos = self.VMrentInfos[self.nextWrf.processDC[task]] + [self.nextWrf.pendingIndexOnDC[task]]

                write_csv_data(self.pkt_trace_file, Workflow_Infos + Task_Infos + VM_Infos)

    # Check whether the machine's lease period needs to be extended
    def extend_remove_VMs(self): 
        expiredVMid = []
        for key in self.VMRemainingTime:
            ind = self.vm_queues_id.index(key)
            self.VMRemainingTime[key] = self.vm_queues[ind].rentEndTime-self.nextTimeStep
            self.VMRemainAvaiTime[key] = self.VMRemainingTime[key] - self.vm_queues[ind].pendingTaskTime

            if self.VMRemainAvaiTime[key] <= 0:
                if self.vm_queues[ind].currentQlen == 0:
                    expiredVMid.append(key) 
                else:
                    while self.VMRemainAvaiTime[key] <= 0:
                        self.VMRemainingTime[key] += self.set.VMpayInterval
                        self.VMRemainAvaiTime[key] += self.set.VMpayInterval
                        self.vm_queues[ind].update_vmRentEndTime(self.set.VMpayInterval)
                        self.update_VMcost(self.vm_queues[ind].loc, self.vm_queues[ind].cpu, True)

        if len(expiredVMid) > 0:
            if not self.nextisUsr:
                nextvmid = self.vm_queues[self.nextUsr].vmid

            for key in expiredVMid:
                del self.VMRemainingTime[key]
                del self.VMRemainAvaiTime[key]
                ind = self.vm_queues_id.index(key)
                del self.vm_queues_id[ind]
                del self.vm_queues_cpu[ind]
                del self.vm_queues_rentEndTime[ind]
                vm = self.vm_queues.pop(ind)
                del self.firstvmWrfLeaveTime[ind]
                del vm        

            # If there is deletion, you need to adjust the index corresponding to self.nextUsr
            if not self.nextisUsr:
                if nextvmid in self.vm_queues_id:
                    self.nextUsr = self.vm_queues_id.index(nextvmid)   
                else:
                    print('wrong')

    # Function prototype is vf_ob, ac_ob, rew, new, _ = env.step(ac)
    def step(self, action):
        # decode the action: the index of the vm

        # ---1) Map & Dispatch
        # maping the action to the vm_id in current VM queue
        diff = action - len(self.vm_queues)
        if diff > -1:
            vmid = self.generate_vmid()
            dcid = 0
            vmtype = diff % self.VMtypeNum 
    
            selectedVM = VM(vmid, self.set.dataset.vmVCPU[vmtype], dcid, self.set.dataset.datacenter[dcid][0], self.nextTimeStep, self.TaskRule)
            self.vm_queues.append(selectedVM)
            self.firstvmWrfLeaveTime.append(selectedVM.get_firstTaskDequeueTime())
            self.vm_queues_id.append(vmid)
            self.vm_queues_cpu.append(self.set.dataset.vmVCPU[vmtype]) 
            self.update_VMcost(dcid, self.set.dataset.vmVCPU[vmtype], True)
            selectedVMind = -1
            self.VMRemainingTime[vmid] = self.set.VMpayInterval
            self.VMRemainAvaiTime[vmid] = self.set.VMpayInterval            
            self.vm_queues[selectedVMind].update_vmRentEndTime(self.set.VMpayInterval)
            self.vm_queues_rentEndTime.append(self.vm_queues[selectedVMind].rentEndTime) 
            self.VMrentInfos[vmid] = [vmid, self.set.dataset.vmVCPU[vmtype],  self.set.dataset.vmPrice[self.set.dataset.vmVCPU[vmtype]], 
                                      self.nextTimeStep, self.vm_queues[selectedVMind].rentEndTime]     
        else:
            selectedVMind = action
        reward = 0
        self.PrenextUsr, self.PrenextTimeStep, self.PrenextTask = self.nextUsr, self.nextTimeStep, self.nextTask 

        # dispatch nextWrf to selectedVM and update the wrfLeaveTime on selectedVM 
        parentTasks = self.nextWrf.get_allpreviousTask(self.PrenextTask)
        if len(parentTasks) == len(self.nextWrf.completeTaskSet(parentTasks)):
            processTime = self.vm_queues[selectedVMind].task_enqueue(self.PrenextTask, self.PrenextTimeStep, self.nextWrf)
            self.VMexecHours += processTime/3600                                                                                                              
            self.firstvmWrfLeaveTime[selectedVMind] = self.vm_queues[selectedVMind].get_firstTaskDequeueTime()
            self.extend_specific_VM(selectedVMind) 

        # ---2) Dequeue nextTask
        if self.isDequeue:
            if self.nextisUsr:
                self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                _, _ = self.usr_queues[self.PrenextUsr].dequeue()
                self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstPktEnqueueTime() 

                self.usr_sent_pktNum[self.PrenextUsr][self.vm_queues[selectedVMind].get_relativeVMloc()] += 1
                self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_appID(), self.nextWrf.get_generateTime()) # record
            else:
                _, _ = self.vm_queues[self.PrenextUsr].task_dequeue()  # Here nextTask actually starts to run
                self.firstvmWrfLeaveTime[self.PrenextUsr] = self.vm_queues[self.PrenextUsr].get_firstTaskDequeueTime()


        # ---3) Update: self.nextTask, and maybe # self.nextWrf, self.finishTask, self.nextUsr, self.nextTimeStep, self.nextisUsr
        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

        if len(temp_Children_finishTask) > 0:
            self.dispatchParallelTaskNum += 1

        while True:

            # self.nextWrf is completed
            while len(temp_Children_finishTask) == 0:
                
                if self.nextisUsr:
                    print('self.nextisUsr maybe wrong')
                _, app = self.vm_queues[self.nextUsr].task_dequeue()  
                self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime()
                if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):
                    respTime = self.nextTimeStep - self.nextWrf.get_generateTime()
                    self.usr_respTime[app.get_originDC()][app.get_appID()] += respTime
                    self.usr_received_wrfNum[app.get_originDC()][app.get_appID()] += 1                    
                    self.completedWF += 1
                    self.remainWrfNum -= 1
                    ddl_penalty = self.calculate_penalty(app, respTime)
                    self.SLApenalty += ddl_penalty
                    self.record_a_completed_workflow(ddl_penalty)
                    del app, self.nextWrf

                self.get_nextTimeStep()
                if self.nextTimeStep is None:
                    break
                self.update_VMRemain_infos()
                self.remove_expired_VMs()                
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

            if self.nextTimeStep is None:
                break

            # Indicates that parallel tasks have not been allocated yet, and len(temp_Children_finishTask)>=1
            if len(temp_Children_finishTask) > self.dispatchParallelTaskNum: 
                to_be_next = None
                while len(temp_Children_finishTask) > self.dispatchParallelTaskNum:
                    temp_nextTask = temp_Children_finishTask[self.dispatchParallelTaskNum]
                    temp_parent_nextTask = self.nextWrf.get_allpreviousTask(temp_nextTask)
                    if len(temp_parent_nextTask) - len(self.nextWrf.completeTaskSet(temp_parent_nextTask)) >0:
                        self.dispatchParallelTaskNum += 1
                    else: 
                        to_be_next = temp_nextTask
                        break

                if to_be_next is not None: 
                    self.nextTask = to_be_next
                    if len(temp_Children_finishTask) - self.dispatchParallelTaskNum > 1:
                        self.isDequeue = False
                    else:
                        self.isDequeue = True
                    break

                else:  # Mainly to loop this part
                    _, _ = self.vm_queues[self.nextUsr].task_dequeue() # Actually start running self.nextTask here
                    self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime()
                    self.get_nextTimeStep() 
                    self.update_VMRemain_infos()
                    self.remove_expired_VMs()                        
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                    self.dispatchParallelTaskNum = 0                     
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)
            else:
                if not self.isDequeue:
                    print('self.isDequeue maybe wrong')      
                self.get_nextTimeStep()
                self.update_VMRemain_infos()
                self.remove_expired_VMs()                    
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.dispatchParallelTaskNum = 0
                if self.nextTimeStep is not None:
                    temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

        self.numTimestep = self.numTimestep + 1
        self.notNormalized_arr_hist = self.stat.update_arrival_rate_history()

        done = False
        if self.remainWrfNum == 0:
            if len(self.firstvmWrfLeaveTime) == 0:
                done = True
            elif self.firstvmWrfLeaveTime[0] == math.inf and self.firstvmWrfLeaveTime.count(self.firstvmWrfLeaveTime[0]) == len(self.firstvmWrfLeaveTime):
                done = True

        if done:
            reward = -self.VMcost-self.SLApenalty
            self.episode_info = {"VM_execHour": self.VMexecHours, "VM_totHour": self.VMrentHours,
                                 "VM_cost": self.VMcost, "SLA_penalty": self.SLApenalty,
                                 "missDeadlineNum": self.missDeadlineNum}
            # print('Useless Allocation has ----> ',self.uselessAllocation)
            self._init()  # cannot delete

        return reward, self.usr_respTime, self.usr_received_wrfNum, self.usr_sent_pktNum, done

    # calculate the total VM cost during an episode
    def update_VMcost(self, dc, cpu, add=True):
        if add:
            temp = 1
        else:
            temp = 0
        self.VMcost += temp * self.set.dataset.vmPrice[cpu]
        self.VMrentHours += temp

    def calculate_penalty(self, app, respTime):
        appID = app.get_appID()
        threshold = app.get_Deadline() - app.get_generateTime()
        if respTime < threshold or round(respTime - threshold, 5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            return 1+self.set.dataset.wsetBeta[appID]*(respTime-threshold)/3600

    def state_info_construct(self):
        '''
        states:
        1.	Number of child tasks: childNum
        2.	Completion ratio: completionRatio
        3.	Workflow arrival rate: arrivalRate (a vector of historical arrivalRate)
        4.	Whether the VM can satisfy the deadline regardless the extra cost: meetDeadline (0:No, 1:Yes)
        5.	Total_overhead_cost = potential vm rental fee + deadline violation penalty: extraCost
        6.  VM_remainTime: after allocation, currentRemainTime - taskExeTime ( + newVMrentPeriod if applicable)
        7.	BestFit - among all the VMs, whether the current one introduces the lowest extra cost? (0, 1)
        '''

        ob = []

        # ---1)task related state:
        childNum = len(self.nextWrf.get_allnextTask(self.nextTask))  # number of child tasks
        completionRatio = self.nextWrf.get_completeTaskNum() / self.nextWrf.get_totNumofTask()  # self.nextWrf: current Wrf
        arrivalRate = np.sum(np.sum(self.notNormalized_arr_hist, axis=0), axis=0)
        task_ob = [childNum, completionRatio]
        task_ob.extend(list(copy.deepcopy(arrivalRate)))

        # calculate the sub-deadline for a task
        if self.nextWrf not in self.appSubDeadline:
            self.appSubDeadline[self.nextWrf] = {}
            deadline = self.nextWrf.get_maxProcessTime()*self.set.dueTimeCoef[self.nextWrf.get_originDC()][self.nextWrf.get_appID()]
            psd = calPSD(self.nextWrf, deadline, self.set.dataset.vmVCPU)
            for key in psd:
                self.appSubDeadline[self.nextWrf][key] = psd[key]+self.nextTimeStep

        # ---2)vm related state:
        for vm_ind in range(len(self.vm_queues)):
            task_est_startTime = self.vm_queues[vm_ind].vmLatestTime()
            task_exe_time = self.nextWrf.get_taskProcessTime(self.nextTask) / self.vm_queues_cpu[vm_ind]
            task_est_finishTime = task_exe_time + task_est_startTime
            temp = round(self.VMRemainingTime[self.vm_queues_id[vm_ind]] - task_est_finishTime, 5)
            if temp > 0:
                extra_VM_hour = 0
                vm_remainTime = temp
            else:
                extra_VM_hour = math.ceil(- temp / self.set.VMpayInterval)
                vm_remainTime = round(self.set.VMpayInterval * extra_VM_hour + self.VMRemainingTime[self.vm_queues_id[vm_ind]] - task_est_finishTime, 5)
            extraCost = (self.set.dataset.datacenter[self.vm_queues[vm_ind].get_relativeVMloc()][-1]) / 2 * self.vm_queues_cpu[vm_ind] * extra_VM_hour

            if task_est_finishTime + self.nextTimeStep < self.appSubDeadline[self.nextWrf][self.nextTask]:
                meetDeadline = 1
            else:
                meetDeadline = 0
                extraCost += 1 + self.set.dataset.wsetBeta[self.nextWrf.get_appID()] * (task_exe_time + self.nextTimeStep - self.appSubDeadline[self.nextWrf][self.nextTask])  # add SLA penalty
            ob.append([])
            ob[-1] = task_ob + [meetDeadline, extraCost, vm_remainTime]

        for dcind in range(self.dcNum):  # for new VM that can be rented
            for cpuNum in self.set.dataset.vmVCPU:
                dc = self.set.dataset.datacenter[dcind]
                task_exe_time = self.nextWrf.get_taskProcessTime(self.nextTask) / cpuNum
                extra_VM_hour = math.ceil(task_exe_time / self.set.VMpayInterval)
                extraCost = dc[-1] / 2 * cpuNum * extra_VM_hour
                if task_exe_time + self.nextTimeStep < self.appSubDeadline[self.nextWrf][self.nextTask]:
                    meetDeadline = 1
                else:
                    meetDeadline = 0
                    extraCost += 1 + self.set.dataset.wsetBeta[self.nextWrf.get_appID()] * (task_exe_time + self.nextTimeStep - self.appSubDeadline[self.nextWrf][self.nextTask])  # add SLA penalty
                vm_remainTime = round(self.set.VMpayInterval * extra_VM_hour - task_exe_time, 5)
                ob.append([])
                ob[-1] = task_ob + [meetDeadline, extraCost, vm_remainTime]

        # if a VM is the best fit, i.e., min(extraCost)
        temp = np.array(ob)
        row_ind = np.where(temp[:, -2] == np.amin(temp[:, -2]))[0]
        bestFit = np.zeros((len(ob), 1))
        bestFit[row_ind, :] = 1
        ob = np.hstack((temp, bestFit))

        # get the dag of current Wrf
        self.nextWrf.app.nodes[self.nextTask]["scheduled"] = 1.0
        for node, data in self.nextWrf.app.nodes(data=True):
            self.nextWrf.app.nodes[node]["sub_deadline"] = self.appSubDeadline[self.nextWrf][node]
            if "scheduled" in data:
                value = self.nextWrf.app.nodes[node]["scheduled"]
                if value is not None and not math.isnan(value):
                    pass
                else:
                    self.nextWrf.app.nodes[node]["scheduled"] = 1.0
            else:
                self.nextWrf.app.nodes[node]["scheduled"] = 0.0

        return ob, self.nextWrf.app, self.nextTask  # state, dag, readyTask_node_id

