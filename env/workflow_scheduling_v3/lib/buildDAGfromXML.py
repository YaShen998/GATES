# assume the runtime in *.xml is the execution time on a CPU with 16 cores
# references: Characterizing and profiling scientific workflows

import xml.etree.ElementTree as ET
# networkx version
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
            # dag.add_node(child.attrib['id'], processTime=float(child.attrib['runtime'])*16, size=size)
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                dag.add_edge(parent, kid)
    return dag, tot_processTime

# # ============ testing =============
# options = {
#     'node_color': 'black',
#     'node_size': 10,
#     'width': 2,
#     'arrowstyle': '-|>',
#     'arrowsize': 10,
# }
# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# sys.path.insert(0, parentdir)
# test = buildGraph('CyberShake', parentdir+'/dax/CyberShake_30.xml')
# nx.draw_networkx(test, arrows=True, **options)
# nx.draw(test)
# plt.show()



# import igraph
# import matplotlib.pyplot as plt
# import xml.etree.ElementTree as ET
#
#
# def buildGraph(type, filename):
#     dag = igraph.Graph(directed=True)
#     dag["type"] = type
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     for child in root:
#         if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
#             size = 0
#             for p in child:
#                 size += int(p.attrib['size'])
#             dag.add_vertex(int(child.attrib['id'][2:]), processTime=float(child.attrib['runtime']) * 16, size=size)
#             # dag.add_node(child.attrib['id'], processTime=float(child.attrib['runtime'])*16, size=size)
#         if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
#             kid = int(child.attrib['ref'][2:])
#             for p in child:
#                 parent = int(p.attrib['ref'][2:])
#                 dag.add_edge(parent, kid)
#     return dag
#
# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# sys.path.insert(0, parentdir)
# test = buildGraph('CyberShake', parentdir+'/dax/CyberShake_30.xml')
# layout = test.layout("kk")
# print(test.vs.indices)
# print(f'Expected output: 3.04, Actual output: {test.vs[1]["processTime"]}')
# # igraph.plot(test, layout=layout)  # not working because require Cairo library https://stackoverflow.com/questions/12072093/python-igraph-plotting-not-available/45416251#45416251

