# import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
# from workflow_scheduling.env.poissonSampling import one_sample_poisson
import math
import heapq


class VM:
    def __init__(self, id, cpu, dcind, abind, t, rule):
        self.vmid = id
        self.cpu = cpu
        self.loc = dcind
        self.abloc = abind
        self.vmQueue = SimQueue()
        self.currentTimeStep = t
        self.rentStartTime = t
        self.rentEndTime = t
        self.processingApp = None
        self.processingtask = None
        self.totalProcessTime = 0
        self.pendingTaskTime = 0
        self.pendingTaskNum = 0
        self.taskSelectRule = rule
        self.currentQlen = 0

    def get_utilization(self, app, task):
        numOfTask = self.totalProcessTime / (app.get_taskProcessTime(task)/self.cpu)
        util = numOfTask/self.get_capacity(app, task) 
        return util

    def get_capacity(self, app, task):
        return 60*60 / (app.get_taskProcessTime(task)/self.cpu)

    def get_vmid(self):
        return self.vmid

    def get_cpu(self):
        return self.cpu

    def get_relativeVMloc(self):
        return self.loc

    def get_absoluteVMloc(self):
        return self.abloc

    ## self-defined
    def cal_priority(self, task, app):
        
        if self.taskSelectRule is None:
            enqueueTime = app.get_enqueueTime(task)
            return enqueueTime
        else:
            task_ExecuteTime_real = app.get_taskProcessTime(task)/self.cpu
            task_WaitingTime = self.get_taskWaitingTime(app, task)
            vm_TotalProcessTime = self.vmQueueTime()
            vm_NumInQueue = self.currentQlen
            task_NumChildren = app.get_NumofSuccessors(task)
            workflow_RemainTaskNum = app.get_totNumofTask() - app.get_completeTaskNum()
            RemainDueTime = app.get_Deadline() - self.currentTimeStep

            priority = self.taskSelectRule(ET = task_ExecuteTime_real, WT = task_WaitingTime, TIQ = vm_TotalProcessTime, 
                    NIQ = vm_NumInQueue, NOC = task_NumChildren, NOR = workflow_RemainTaskNum, RDL= RemainDueTime)
            return priority


    def get_firstTaskEnqueueTimeinVM(self):
        if self.processingApp is None:
            return math.inf
        return self.processingApp.get_enqueueTime(self.processingtask)

    def get_firstTaskDequeueTime(self):
        if self.get_pendingTaskNum() > 0:
            return self.currentTimeStep
        else:
            return math.inf

    def get_firstDequeueTask(self):
        return self.processingApp, self.processingtask

    def get_pendingTaskNum(self):
        if self.processingApp is None:
            return 0
        else:
            return self.vmQueue.qlen()+1  # 1 is needed

    def task_enqueue(self, task, enqueueTime, app, resort=False):
        temp = app.get_taskProcessTime(task)/self.cpu
        self.totalProcessTime += temp
        self.pendingTaskTime += temp        
        self.currentQlen = self.get_pendingTaskNum()

        app.update_executeTime(temp, task)
        app.update_enqueueTime(enqueueTime, task, self.vmid)
        self.vmQueue.enqueue(app, enqueueTime, task, self.vmid, enqueueTime)

        if self.processingApp is None:
            self.process_task()

        return temp

    def task_dequeue(self, resort=True):
        task, app = self.processingtask, self.processingApp

        # self.currentTimeStep always == dequeueTime(env.nextTimeStep)

        qlen = self.vmQueue.qlen()
        if qlen == 0:
            self.processingApp = None
            self.processingtask = None
        else:
            if resort:  
                tempvmQueue = SimQueue()
                for _ in range(qlen):
                    oldtask, oldapp = self.vmQueue.dequeue()
                    priority = self.cal_priority(oldtask, oldapp)
                    heapq.heappush(tempvmQueue.queue, (priority, oldtask, oldapp))
                self.vmQueue.queue = tempvmQueue.queue

            self.process_task()
            self.currentQlen-=1

        return task, app 

    def process_task(self): #
        self.processingtask, self.processingApp = self.vmQueue.dequeue()
        enqueueTime = self.processingApp.get_enqueueTime(self.processingtask)
        processTime = self.processingApp.get_executeTime(self.processingtask)

        taskStratTime = max(enqueueTime , self.currentTimeStep)
        leaveTime = taskStratTime +processTime

        self.processingApp.update_enqueueTime(taskStratTime, self.processingtask, self.vmid)
        self.pendingTaskTime -= processTime
        self.processingApp.update_pendingIndexVM(self.processingtask, self.pendingTaskNum)
        self.pendingTaskNum+=1
        self.currentTimeStep = leaveTime

    def vmQueueTime(self): 
        return max(round(self.pendingTaskTime,3), 0)

    def vmTotalTime(self): 
        return self.totalProcessTime
    
    def vmLatestTime(self): 
        # return self.totalProcessTime+self.rentStartTime    
        return self.currentTimeStep + self.pendingTaskTime
    
    def get_vmRentEndTime(self):
        return self.rentEndTime
    
    def update_vmRentEndTime(self, time):
        self.rentEndTime += time

    def get_taskWaitingTime(self, app, task): 
        waitingTime = self.currentTimeStep - app.get_enqueueTime(task)
        return waitingTime
