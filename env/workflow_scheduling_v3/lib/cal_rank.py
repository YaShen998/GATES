import networkx as nx


def calcPURank(wf, vmVCPU, theta=0):
    """Calculate the probabilistic upward rank of all tasks using Eq(11) in the paper:
    Deadline-constrained Probabilistic List Scheduling (ProLiS) -- A static scheduling heuristic

      Deadline-Constrained Cost Optimization Approaches for Workflow Scheduling in Clouds
                                by
             Quanwang Wu, Fuyuki Ishikawa, Qingsheng Zhu, Yunni Xia, Junhao Wen
     IEEE Transactions on Parallel and Distributed Systems 2017

    However, since we do not consider data transmission cost in our paper, the implementation is the same as Eq(9)

    Args:
        wf (class object): A workflow object generated from env/application.py
        theta (int): a larger theta indicates the tendency to focus on the transmission time is strong in Eq(12)

    Returns:
        PURank (dict): The probabilisti upward ranks of all tasks in the workflow

    Examples:
        # # test for function: calcPURank()
        from workflow_scheduling.env.application import Application
        from workflow_scheduling.env.get_DAGlongestPath import get_longestPath_nodeWeighted
        workflowType = 'CyberShake'
        sdict = {'CyberShake': {'E': 300, 'S': 600, 'ZS': 600, 'P': 3, 'ZP': 433}}
        testWF = nx.DiGraph(type=workflowType)
        wsetProcessTime = 0
        for i in range(6):
            if i < 3:
                service = 'E'
            else:
                service = 'P'
            testWF.add_node(i, processTime=sdict[workflowType][service] * 16, size=1000)

        # # WF1: sequential, Output: {0: 303.0, 1: 203.0, 2: 103.0, 3: 3.0, 4: 2.0, 5: 1.0}
        # testWF.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5)])

        # # WF2: parallel, Output: {0: 202.0, 1: 102.0, 2: 101.0, 3: 1.0, 4: 2.0, 5: 1.0}
        testWF.add_edges_from([(0, 1), (0, 2), (2, 3), (1, 4), (4, 5)])
        appID = 0
        app = Application(0.1, testWF, appID, 0, get_longestPath_nodeWeighted(testWF))
        print(calcPURank(app))

    """

    rankuDict = {}

    tasks = wf.app.nodes  # get the list of all tasks in the wf
    exit = [n for n, d in wf.app.out_degree() if d == 0]  # get the exit task ID

    for i in tasks:
        cost = 0
        for leaf in exit:
            if i is leaf:
                cost = tasks[i]['processTime'] / vmVCPU[-1]  # dataset.vmVCPU[-1] is the fastest vm
            else:
                for path in nx.all_simple_paths(wf.app, source=i, target=leaf):
                    temp = 0
                    for node in path:
                        temp += 1.0 * tasks[node]['processTime'] / vmVCPU[-1]
                    if temp > cost:
                        cost = temp
        rankuDict[i] = cost

    return rankuDict


def calPSD(wf, deadline, vmVCPU):
    """Calculate the deadline distribution based on probabilistic upward rank of all tasks using Eq(13) in the paper
    However, since we do not consider data transmission cost in our paper, the implementation is the same as Eq(10)

    Args:
        wf (class object): A workflow object generated from env/application.py
        deadline (float): the maximum execution time for a workflow, i.e., the result of the workflow should be returned within "deadline"

    Returns:
        psd (dict): The deadline distribution of all tasks in the workflow

    Examples:
        # # test for function: calPSD()
        from workflow_scheduling.env.application import Application
        from workflow_scheduling.env.get_DAGlongestPath import get_longestPath_nodeWeighted
        workflowType = 'CyberShake'
        sdict = {'CyberShake': {'E': 300, 'S': 600, 'ZS': 600, 'P': 3, 'ZP': 433}}
        testWF = nx.DiGraph(type=workflowType)
        wsetProcessTime = 0
        for i in range(6):
            if i < 3:
                service = 'E'
            else:
                service = 'P'
            testWF.add_node(i, processTime=sdict[workflowType][service] * 16, size=1000)

        # # WF1: sequential, Output: {0: 200.0, 1: 400.0, 2: 600.0, 3: 602.0, 4: 604.0, 5: 606.0}
        # testWF.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5)])

        # # WF2: parallel, Output: {0: 300.0, 1: 600.0, 2: 603.0, 3: 606.0, 4: 603.0, 5: 606.0}
        testWF.add_edges_from([(0, 1), (0, 2), (2, 3), (1, 4), (4, 5)])
        appID = 0
        app = Application(0.1, testWF, appID, 0, get_longestPath_nodeWeighted(testWF))
        print(calPSD(app, 606))

    """
    psd = {}
    rankuDict = calcPURank(wf, vmVCPU)
    # entrance = [n for n, d in wf.app.in_degree() if d == 0]  # get the entrance task ID
    PURankEntry = max(rankuDict.values())

    for task in rankuDict:
        psd[task] = deadline * (PURankEntry - rankuDict[task] + wf.get_taskProcessTime(task) / vmVCPU[-1]) / PURankEntry
    return psd

