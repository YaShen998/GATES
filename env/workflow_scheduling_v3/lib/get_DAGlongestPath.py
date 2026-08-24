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
