# DAX runtimes assume execution on a 16-core CPU.
# Reference: Characterizing and Profiling Scientific Workflows.
import xml.etree.ElementTree as ET

import networkx as nx


def buildGraph(type, filename):
    tot_processTime = 0
    dag = nx.DiGraph(type=type)
    with open(filename, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        xml_file.close()
    root = tree.getroot()
    for child in root:
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
            size = 0
            for p in child:
                size += int(p.attrib['size'])
            dag.add_node(int(child.attrib['id'][2:]), processTime=float(child.attrib['runtime']) * 16, size=size)
            tot_processTime += float(child.attrib['runtime']) * 16

        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                dag.add_edge(parent, kid)
    return dag, tot_processTime
