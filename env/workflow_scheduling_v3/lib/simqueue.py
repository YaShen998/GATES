import logging
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
import heapq
import math

class SimQueue:
    def __init__(self):
        self.queue = []

    def qlen(self):
        return len(self.queue)

    def enqueue(self, pkt, t, task, vmID, priority):
        if task is None:
            pkt.update_enqueueTime(t, task, vmID)
        heapq.heappush(self.queue, (priority, task, pkt))

    def dequeue(self):
        if len(self.queue) > 0:
            _, task, pkt = heapq.heappop(self.queue)
            return task, pkt
        else:
            logging.error("queue is empty")
            sys.exit(1)

    def getFirstPktEnqueueTime(self):
        if len(self.queue) > 0:
            _, task, firstPkt = self.queue[0]
            enqueueTime = firstPkt.get_readyTime(task)
            return enqueueTime
        else:
            return math.inf

    def getFirstPkt(self):
        if len(self.queue) > 0:
            _, task, firstPkt = self.queue[0]

            return firstPkt, task
        else:
            return None, None

