# # networkx version
import networkx as nx

def get_longestPath_nodeWeighted(G):
    entrance = [n for n, d in G.in_degree() if d == 0]
    exit = [n for n, d in G.out_degree() if d == 0]
    cost = 0
    for root in entrance:
        for leaf in exit:
            for path in nx.all_simple_paths(G, source=root, target=leaf):
                temp = 0
                for node in path:
                    temp += G.nodes[node]['processTime']
                if temp > cost:
                    cost = temp
    return cost




# # igraph version
# import igraph
#
# def get_longestPath_nodeWeighted(G):
#     vertices = G.vs.indices
#     entrance = []
#     exit = []
#     for v in vertices:
#         if G.degree(v, mode='IN') == 0:
#             entrance.append(v)
#         if G.degree(v, mode='OUT') == 0:
#             exit.append(v)
#     cost = 0
#     for root in entrance:
#         for leaf in exit:
#             for path in G.get_all_simple_paths(root, to=leaf, mode='OUT'):
#                 temp = 0
#                 for node in path:
#                     temp += G.vs[node]['processTime']
#                 if temp > cost:
#                     cost = temp
#     return cost
