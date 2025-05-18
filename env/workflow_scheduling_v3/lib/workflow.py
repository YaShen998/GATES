
class Workflow:
    def __init__(self, generateTime1, app, appID, originDC, time, ratio, index):
        self.generateTime = generateTime1
        self.origin = originDC
        self.app = app
        self.appID = appID
        self.appArivalIndex = index
        self.readyTime = {}
        self.enqueueTime = {}
        self.dequeueTime = {}
        self.pendingIndexOnDC ={}
        self.processDC = {}
        self.executeTime = {}
        self.queuingTask = []
        self.maxProcessTime = time
        self.dueTimeCoef = ratio
        self.deadlineTime = self.maxProcessTime * self.dueTimeCoef + self.generateTime

    def __lt__(self, other):  
        return self.generateTime < other.generateTime

    def get_completeTaskNum(self):
        return len(self.dequeueTime)

    def get_maxProcessTime(self):
        return self.maxProcessTime

    def is_completeTaskSet(self, s_list):
        if set(s_list).issubset(set(self.dequeueTime.keys())):
            return True
        else:
            return False

    def completeTaskSet(self, s_list):
        return set(s_list).intersection(set(self.dequeueTime.keys()))

    def update_pendingIndexVM(self, task, pendingTndex):
        self.pendingIndexOnDC[task] = pendingTndex

    def update_enqueueTime(self, time, task, vmID):
        if task not in self.enqueueTime:
            self.readyTime[task] = time
            self.enqueueTime[task] = time
            self.processDC[task] = vmID  

        elif time > self.enqueueTime[task]:
            self.enqueueTime[task] = time

    def update_dequeueTime(self, time, task):
        self.dequeueTime[task] = time

    def update_executeTime(self, time, task):
        self.executeTime[task] = time

    def get_executeTime(self, task):
        return self.executeTime[task]

    def add_queuingTask(self, task):
        self.queuingTask.append(task)

    def remove_queuingTask(self, task):
        self.queuingTask.remove(task)

    def get_taskDC(self, task):
        if task in self.processDC:
            return self.processDC[task]
        else:
            return None

    def get_generateTime(self):
        return self.generateTime

    def get_enqueueTime(self, task):
        return self.enqueueTime[task]

    def get_readyTime(self, task):
        return self.readyTime[task]

    def get_originDC(self):
        return self.origin

    def get_appID(self):
        return self.appID

    def get_taskProcessTime(self, task):
        return self.app.nodes[task]['processTime']  # networkx
        # return self.app.vs[task]['processTime']  # igraph

    # return the next task list given the current task ID
    def get_allnextTask(self, task):
        if task is None:
            root = [n for n, d in self.app.in_degree() if d == 0]  # networkx
            return root
        else:
            return list(self.app.successors(task))

    def get_NumofSuccessors(self, task):          
        node_succ = self.get_allnextTask(task)
        return len(node_succ)

    def get_Deadline(self):              
        return self.deadlineTime 
 
    # return the parent tasks given the current task
    def get_allpreviousTask(self, task):
        if task is None:
            return []
        else:
            return list(self.app.predecessors(task))

    def get_totNumofTask(self):
        return self.app.number_of_nodes()  # networkx
        # return self.app.vcount()    # igraph

    def get_allTask(self):
        return list(self.app.nodes)  # networkx
        # return self.app.vs.indices   # igraph
