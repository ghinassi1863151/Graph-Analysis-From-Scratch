#############################################################################    
############################## IMPORTS ######################################
#############################################################################

from numpy.core.fromnumeric import product
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
import pickle
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool as ThreadPool
import itertools
import random
import math
import json
import heapq as heap
from collections import defaultdict, Counter, OrderedDict, deque
from termcolor import colored
from beautifultable import BeautifulTable
import matplotlib.pyplot as plt
from matplotlib import style 
import networkx as nx
from datetime import timedelta  
# Import packages for data visualization

from matplotlib.pyplot import close, figure
    

plt.style.use('dark_background')


def save_pickle(element, path):
    with open(f"{path}", 'wb') as f:
        pickle.dump(element, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(f"{path}", 'rb',) as f:
        return pickle.load(f)
    
#############################################################################    
############################## GLOBAL VARIABLES #############################
#############################################################################

# dataframes
a2q_path = './graphs/a2q.pickle'
c2a_path = './graphs/c2a.pickle'
c2q_path = './graphs/c2q.pickle'
graph_path = 'graphs/weighted_graph.json'


# GLOBAL VARIABLES
weights_global = [1, 0.4, 0.2]
source_global = 'replier'
destination_global = 'questioner'

def loading_data():
    
    a2q = load_pickle(a2q_path)
    c2a = load_pickle(c2a_path)
    c2q = load_pickle(c2q_path)

    graph = json.loads(open(graph_path).read())
    
    return a2q, c2a, c2q, graph


a2q, c2a, c2q, graph_tot  = loading_data()

#############################################################################    
############################ BUILDING THE GRAPH #############################
#############################################################################

def subset_with_parsing(dfrom, dto, column, *dataframes):
    '''
    this function selects a subset of rows from the dataframe given an interval of time and parse the given column
    parameters:
    dfrom <- starting date
    dto <- end date
    column <- the column to apply the constraint
    dataframes <- the k dataframes we want to preprocess
    '''
    dataframes = list(dataframes)
    for df in dataframes:   
        df = df[(df[column] > dfrom) & (df[column] < dto)]
        df = df[column].apply(lambda el: int(datetime.fromtimestamp(int(el)).strftime('%Y%m%d')))
    
    return dataframes
    
    
# FUNCTIONS TO BUILD THE GRAPH

def add_to_weightedGraph(source, destination, weight, graph):
    '''
    this function adds an edge to a weighted graph
    
    parameters:
    source <- the exit node
    destination <- the entry node
    weight <- the weight of the edge
    dictionary <- the initial graph
    '''
    
    if source not in graph :
        graph[source] = defaultdict(int)
    graph[source][destination] += weight
    

def create_weighted_graph(weights, source, destination, *dataframes):
    '''
    weights: a vector of weigths assigned at each type of link
    source: the vector of the source nodes
    destination: the vector of the destinations nodes
    dataframes: the dataframes from which we select the nodes
    '''
    dataframes = list(dataframes)
    graph_dictionary = {}
    
    for i in tqdm(range(len(dataframes))):
    
        for row in zip(dataframes[i][source], dataframes[i][destination]): 
            add_to_weightedGraph(str(row[0]), str(row[1]), weights[i], graph_dictionary)
        
    return graph_dictionary


def add_toGraph(source, destination, dictionary):
    '''
    this function adds an edge to a graph
    
    parameters:
    source <- the exit node
    destination <- the entry node
    dictionary <- the initial graph
    '''
    if source not in dictionary:
        dictionary[source] = set()
        
    dictionary[source].add(destination)
    

def create_graph(source, destination, *dataframes):
    
    dataframes = list(dataframes)
    out = []
    
    for df in tqdm(dataframes):
        
        graph_dictionary = {}

        for row in zip(df[source], df[destination]): 
            add_toGraph(row[0], row[1], graph_dictionary)
        
        out.append(graph_dictionary)
    
    return out

def create_weighted_graph_interval(dfrom, dto, weights, source, destination, column, *dataframes):
    
    '''
    weights: a vector of weigths assigned at each type of link
    source: the vector of the source nodes
    destination: the vector of the destinations nodes
    dataframes: the dataframes from which we select the nodes
    '''
    
    graph_dictionary = {}
    dataframes = list(dataframes)
    
    for i in range(len(dataframes)):   
        dataframes[i] = dataframes[i][(dataframes[i][column] >= dfrom) & (dataframes[i][column] <= dto)]

        for row in zip(dataframes[i][source], dataframes[i][destination]): 
            add_to_weightedGraph(str(row[0]), str(row[1]), weights[i], graph_dictionary)
    
    
    return graph_dictionary



#############################################################################    
############################ FUNCTIONALITY 1 ################################
#############################################################################


def check_directed(graph_sources, graph_destinations):
    '''
    Check if a graph is directed or not
    '''
    if any(dest not in graph_sources for dest in graph_destinations):
        return True
    # check for back routing edges, if there is at least one element
    if any(source not in graph_destinations for source in graph_sources):
        return True
    return False

def functionality1(graph):
    '''
    our functionality 1, extracts all requested values from [graph]
    '''
    degree_freq = graph['questioner'].value_counts()
    
    sources = np.array(graph.iloc[:, 0])
    destinations = np.array(graph.iloc[:, 1])
    
    out = check_directed(sources, destinations)
    edges = len(sources)
    
    users = set(sources).union(set(destinations))
    
    nodes = len(users)
    
    average_link = np.round(edges/nodes, 2)
    
    if out: # checking if the graph is directed
        
        # if a graph is directed the density is defined as: |E| / |V|*(|V|-1)
        density = edges/(nodes*(nodes-1)) 
    
    else:
        
        # if a graph is undirected: 2*|E| / |V|*(|V|-1)
        density = (2*edges)/(nodes*(nodes-1))
    
    return visualization1(out, nodes, edges, average_link, density, degree_freq)


# VISUALIZATION 1

def visualization1(directed, nodes, edges, average_link, density, degree_freq):
    '''
    Our feature1 plot!
    this computes every requested feature and returns a table visualization!
    '''
    table = BeautifulTable()
    table.set_style(BeautifulTable.STYLE_BOX_DOUBLED)

    table.rows.append(['Directed', colored('True' if directed else 'False', 'green' if directed else 'red')])
    table.rows.append(['Number of users', nodes])
    table.rows.append(['Number of answers/comments', edges])
    table.rows.append(['Average number of links per user', round(average_link, 4)])
    table.rows.append(['Density degree', density])
    table.rows.append(['Dense', colored('True' if (density > 0.5) else 'False', 'green' if (density > 0.5) else 'red')])

    print(table)
    
    degrees = range(len(degree_freq))
    plt.figure(figsize=(15, 10)) 
    plt.loglog(degrees, degree_freq,'-ok', alpha = 0.8, markersize=5, linewidth=0.5,
             markerfacecolor='m', markeredgecolor='cyan', markeredgewidth=0.5) 

    plt.xlabel('Degree');
    plt.ylabel('Frequency');
    
    
############################################################################# ############################ FUNCTIONALITY 2 ################################
#############################################################################


def findAllShortestPaths(G, start, end, min_path_cost):
    '''
    This takes an input Graph and computes every best path that goes from [start] to [end]
    '''
    
    
    # Initialize the queue, final list of paths and weigths variable
    q = deque()
    
    final = []
    weights = 0
    
    # Initialize the path
    temp_path = [start]
    
    q.append(temp_path)
    
    # Iterating until the queue is empty 
    while q:
        tmp_path = q.popleft()
        
        last_node = tmp_path[-1]
    
        # If the last node is our end node --> check the sum weights and if it is equal to minimum Dijkstra distance --> this is a correct minimum path 
        if last_node == end:
            for i in range(len(tmp_path)-1):
                weights += G[tmp_path[i]][tmp_path[i+1]]
            
            if weights == min_path_cost:
                final.append(tmp_path)
                
        # Enqueue new path to analyze in the queue
        if (last_node in G):
            for node in G[last_node]:
                if node not in tmp_path:
                    new_path = [*tmp_path, node]
                    q.append(new_path)
                
    return final

def closeness(graph, user):
    
    _, pathsWeightDict = dijkstra(graph, user)
    pathsTotalCost = 0
    
    for source in pathsWeightDict.keys():
        
        if pathsWeightDict[source] != float('inf'):
            pathsTotalCost += pathsWeightDict[source]
            
    return len(graph.keys())/pathsTotalCost

def degreeCentrality(graph, user, degree = 'out'):
    
    if degree == 'out':
        if user in graph.keys():
            outdegree = len(graph[user]) # degree-out
        else:
            outdegree = 0
            
        return outdegree
    
    if degree == 'in':
        
        indegree = 0
        for source in graph.keys():
            if source != user:
                if user in graph[source].keys():
                    indegree += 1 # degree-in
        return indegree


def compute_pagerank(G, user, alpha, pagerank_up):
    
    '''
    G: graph as dict of dicts
    user: the user we are interested in
    alpha: the teleport probability
    pagerank_up: a dictionary with the previous page rank scores to be updated
    '''
    n = len(G.keys())
    # find parents of the node
    parents = [key for key in G.keys() if ((user != key) and (user in G[key]))]
    
    # this term is the sum of the pagerank of a parent divided by his outdegree (the score has to be shared between the childrens)
    summation_pr = sum((pagerank_up[parent]/f.degreeCentrality(G, parent)) for parent in parents) # default is on outdegree
    
    # update page rank
    pagerank_up[user] = alpha/n + (1-alpha)*summation_pr
    
    return pagerank_up[user]
    
def pagerank(G, user, alpha, iterations = 100):
    
    users = list(G.keys())
    
    # create the starting page rank dict, the initial value will be the indegree
    pagerank_dict = {node: f.degreeCentrality(G, node, degree='in') for node in users}
    
    i = 0
    while i < iterations:
        
        # for each node update his score iteratively
        for node in users:
            pagerank_dict[node] = compute_pagerank(G, node, alpha, pagerank_dict)
         
        i+=1

    return pagerank_dict[user]

def betweeness(G, node):
    
    # Create a list with all the nodes except the one considered
    nodes = list(G.keys())
    if node not in nodes :
        return 0
    
    nodes.remove(node)
    
    # Iterating the list with a double for-loop
    res = 0
    for u in nodes :
        for w in nodes:
            paths_total = 0
            paths_with_node = 0
            # Take lenght of the shortest path between u and w
            parents, costs = dijkstra(G, u)
            min_path_cost = costs[w]
            
            # All the min paths
            min_paths = findAllShortestPaths(G, u, w, min_path_cost)
    
            # Number of paths that contains v between u and w
            for current_min_path in min_paths :
                if node in current_min_path :
                    paths_with_node += 1
                    
        res += paths_with_node / len(min_paths)
    
    return res

def functionality2(dfrom, dto, user, metric, plot = True):
    '''
    takes as param [metric] this possible values :
    - closeness
    - degree
    - page
    - betweenness
    '''
    G = create_weighted_graph_interval(dfrom, dto, weights_global, source_global, destination_global, 'date', a2q, c2a, c2q)
    
    if metric == 'closeness':
        result= closeness(G, user)
    
    if metric == 'degree':
        result= degreeCentrality(G, user)
    
    if metric == 'page':
        result= pagerank(G, user)
    
    if metric == 'betweenness':
        
        '''
        for performance reasons, we decided to run a sampling procedure on our graph
        to extract a sample that contains the node that the user has provided.
        Betweeness will then be run on the sampled graph
        '''
        G_sampled = {}
        count = 0
        while user not in G_sampled:
            keys = random.sample(list(G), 800)
            count += 1
            values = [G[k] for k in keys]
            G_sampled = dict(zip(keys, values))

        result = betweeness(G_sampled, user)
        
    if plot:
        start = datetime.strptime(str(dfrom),'%Y%m%d').date()
        dates = [int(''.join(str(start + timedelta(days=i)).split('-'))) for i in range(5)]
        return visualization2(result, user, metric, dates)

    return result


# VISUALIZATION 2

def plot_neighbours(user):
    '''
    This plots a given [user] neighbors
    '''

    G_nx = nx.DiGraph()
    for key in random.sample(graph_tot[user].keys(), 15):
        G_nx.add_edge(user, key)
        G_nx[user][key]['weight'] = graph_tot[user][key]
        if key in graph_tot.keys():
            if (len(graph_tot[key].keys()) < 5) and (len(graph_tot[key].keys()) > 0):
                for el in graph_tot[key].keys():
                    G_nx.add_edge(key, el)
                    G_nx[key][el]['weight'] = graph_tot[key][el]

            else:
                for el in random.sample(graph_tot[key].keys(), 5):
                    G_nx.add_edge(key, el)
                    G_nx[key][el]['weight'] = graph_tot[key][el]

    colors = ['darkcyan' if node == user else 'white' for node in G_nx]
    sizes = [1000 if node == user else 400 for node in G_nx]

    weights = [G_nx[u][v]['weight'] for u,v in G_nx.edges]   
    plt.figure(figsize=(16, 8))
    pos = nx.spring_layout(G_nx)

    ax = plt.gca()
    ax.margins(0.03)
    plt.axis("off")
    plt.title(f'Network of the user {user}', fontsize = 22)
    box = dict(boxstyle='round', facecolor='snow', alpha=0.7, edgecolor="grey")
    plt.tight_layout()

    nx.draw_networkx_nodes(G_nx, pos, node_size=400, node_color='white', 
                         alpha = .3, label = 'Neighbors of input node');

    nx.draw_networkx_nodes(G_nx, pos, node_size=1000, nodelist = [user], 
                         node_color='crimson', label = 'Input node: {}'.format(user));

    nx.draw_networkx_edges(G_nx, pos, edgelist=G_nx.edges, width=weights, alpha = 0.8, edge_color="steelblue", 
                         arrowsize=15, connectionstyle='arc3,rad=0.05');
    
def metric_evo(user, dates, metric):
    
    values = []

    for i in range(len(dates)-1):
        values.append(functionality2(dates[i], dates[i+1], user, metric, plot = False))

    plt.figure(figsize=(20, 8))
    plt.plot(dates[1:], values)
    plt.xlabel('days')
    plt.ylabel(metric)
    plt.title(f'{metric} evolution for user {user}');


def visualization2(result, user, metric, dates):
    '''
    takes result from functionality 2, plots user neighbors and result
    '''
    
    print(f"Result for user's {user} {metric} : {result}")
    plot_neighbours(user)
    metric_evo(user, dates, metric)
    
    
############################################################################# ############################ FUNCTIONALITY 3 ################################
#############################################################################

def dijkstra(G, startingNode):
    '''
    Dijkstra alorithm implementation for finding best_path
    '''
    # set of visited Nodes
    visited = set()
    # dictionary of parent nodes
    parents_dict = {}
    # define a heap of possible path choices
    paths = []
    # node costs
    costs = defaultdict(lambda: float('inf'))
    costs[startingNode] = 0
    heap.heappush(paths, (0, startingNode))
    depth = 0
    while paths:
        # proceed by popping the shorter cost node
        _, node = heap.heappop(paths)
        # set current node as visited
        visited.add(node)

        if(node not in G):
            continue
        # for every adjacent node of the currently selected path, update the path cost and pop it!
        for adjacentNode, weight in G[node].items():
            if adjacentNode in visited:	continue

            newCost = costs[node] + weight
            # see if this node has already a cost registered, if not, update the paths!
            if costs[adjacentNode] > newCost:
                parents_dict[adjacentNode] = node
                costs[adjacentNode] = newCost
                heap.heappush(paths, (newCost, adjacentNode))
        depth += 1
    return parents_dict, costs

def functionality3(dfrom, dto, source, dest, node_paths):
    '''
    Our functionality 3 implementation! This finds the best path in the graph that reaches [dest]
    from a node and extracts both the path and its weights.
    '''
    G = create_weighted_graph_interval(dfrom, dto, weights_global, source_global, destination_global, 'date', a2q, c2a, c2q)
    
    # prepare for iteration
    nodes = node_paths
    nodes.insert(0, source)
    nodes.append(dest)
    # initialize output values
    final_path = []
    final_cost = 0
    # start iterating
    for i in range(len(nodes) -1 ):
        # update subpath
        current_node = nodes[i]
        dest_node = nodes[i+1]
        # find best subpath
        parents_dict, costs = dijkstra(G, current_node)
        curr_path = []

        # reconstruct path
        navigating = True
        navigating_node = dest_node
        navigating_weight = 0
        
        
        while navigating :
            curr_path.insert(0, navigating_node)
            if(navigating_node not in parents_dict):
                navigating = False
            else :
                navigating_node = parents_dict[navigating_node]

        final_path.extend([p for p in curr_path if p not in final_path])
    
        if(costs[dest_node] != float('inf')):
            
            final_cost += costs[dest_node] 
        else :
            print("Impossible, no existing path")
            return final_path, float('inf')
    
    print(f"Best path is {final_path} with cost {final_cost}")
    print('-'*75, '\n')
    
    weights = []
    
    for i in range(len(final_path) -1):
        weights.append(G[final_path[i]][final_path[i+1]])
        
    return visualization3(final_path, weights, G)


def visualization3(l, weights, graph):
    '''
    Plots a graph best path and some of its neighbors,
    highliting both the best path and its nodes
    
    Params:
        [l] : best path
        [weights] : list of weights of edges involved in the best path
        [graph] : input graph
    '''
    
    # create NX graph
    plt.figure(figsize=(15, 10), dpi=100)
    G = nx.Graph()
    G_sub = G.copy()
    # add every node to the graph
    G.add_nodes_from(l)
    # generate edges for path nodes
    
    edges = create_edges(l, weights)
    
    # add adges to main graph
    for edge in edges :
        G.add_edge(edge[0], edge[1], weight = edge[2], color = 'cyan')
        
    for node in l:
        added = list(graph[node].keys())[:10]
        G_sub.add_weighted_edges_from(add_extra(added,node, graph))
        
    seed = 5587504
    pos2 = nx.spring_layout(G_sub, seed = seed)
    
    nx.draw_networkx_nodes(G_sub, pos2, node_size=400, 
                         node_color='white',)
    nx.draw_networkx_nodes(G, pos2, node_size=1000, 
                         node_color='crimson',)
    
    # draw first edges
    nx.draw_networkx_edges(G_sub, pos2, width=1, alpha=0.6, edge_color = 'deepskyblue')
    for edge in nx.edges(G) :
        nx.draw_networkx_edges(G, pos2, width=5 * G[edge[0]][edge[1]]['weight'], edgelist = [(edge[0], edge[1])], alpha = 0.8,
                               edge_color=G[edge[0]][edge[1]]['color'], 
                                arrowsize=15, connectionstyle='arc3,rad=0.05')
    
    plt.title(f'Shortest path visualization');
    plt.show()



############################################################################# ############################ FUNCTIONALITY 4 ################################
#############################################################################


# FUNCTIONALITY 4

def extract_time_interval_nodes(df):
    # extract source nodes in that time interval
    time_node_dictionary = defaultdict(set)
    # first lets compute the users that appear in which timeframes
    for row in zip(df['replier'], df['questioner'], df['date']): 
        time_node_dictionary[row[2]].add((row[0], row[1]))
    return time_node_dictionary

def find_nodes_in_time_interval(graph, start, end) : 
    ''' finds nodes in [graph] in time interval [start] - [end]'''
    nodes = []
    for key,value in graph.items():
        if(key > start and key < end and value not in nodes):
            nodes.append(value)
    return nodes 

def build_graph_from_df(nodes_df, source_graph):
    output = defaultdict(lambda : defaultdict(int))
    for source, dest in zip(nodes_df['replier'], nodes_df['questioner']) :
        output[str(source)][str(dest)] = source_graph[str(source)][str(dest)]
    return output

def bfs(G, s, t, parent): 
    '''
    Returns true if there is a path from 
    source 's' to sink 't' in 
    residual Graph. Also fills 
    parent[] to store the path '''
    # Mark all the vertices as not visited 
    nodes = nx.nodes(G)
    visited = defaultdict(bool)

    # Create a queue for BFS 
    queue=[] 

    # Mark the source node as visited and enqueue it 
    visited[s] = True
    queue.append(s)
    
    # Standard BFS Loop 
    while queue: 
        #Dequeue a vertex from queue and print it 
        node = queue.pop(0) 
        # Get all adjacent vertices of 
        # the dequeued vertex u 
        # If a adjacent has not been
        # visited, then mark it 
        # visited and enqueue it 
        if(node not in nodes):
            continue

        for idx in G[node]:
            if (not visited[idx] and G[node][idx]['weight'] > 0) : 
                queue.append(idx) 
                visited[idx] = True
                parent[idx] = node
            
    # If we reached sink in BFS starting
    # from source, then return 
    # true, else false 
    return visited[t]

# Function for Depth first search 
# Traversal of the graph
def dfs(G,s,visited):
    visited[s]=True
    if s not in G :
        return
    for node in G[s]:
        if G[s][node]["weight"] > 0 and not visited[node]:
            dfs(G,node,visited)


# Returns the min-cut of the given graph 
def minCut(G, source, sink): 
    graph = G.copy()
    nodes = nx.nodes(G)
    # This is filled by BFS and used to store path 
    parent = {key: -1 for key in nx.nodes(G)}
    max_flow = 0 # There is no flow initially 
    # Augment the flow while there is path from source to sink 

    s = sink
    while(bfs(graph, source, sink, parent)) : 
        # Find minimum residual capacity of the edges along the 
        # path filled by BFS. Or we can say find the maximum flow 
        # through the path found. 
        path_flow = float("inf") 
        
        s = sink
        while(s != source): 
            path_flow = min(path_flow, G[parent[s]][s]['weight']) 
            s = parent[s] 
            
        # Add path flow to overall flow 
        max_flow += path_flow 
        
        v = sink 
        # update residual capacities of the edges and reverse edges 
        # along the path 
        
        while(v != source): 
            u = parent[v] 
            graph[u][v]["weight"] -= path_flow 
            
            if(v in nodes and u in graph[v]):
                graph[v][u]["weight"] += path_flow 
                
            v = parent[v] 
    
    visited ={key: False for key in nodes}
    dfs(graph,s,visited)

    # print the edges which initially had weights
    # but now have 0 weight
    edge_to_remove = []
    weight = 0
    for i in nodes:
        for j in G[i]:
            if (graph[i][j]["weight"] == 0) and (visited[i]):
                edge_to_remove.append((i, j))
                weight += G[i][j]["weight"]
                
    return edge_to_remove, weight

def random_nodes(G1_nodes, G2_nodes):
    '''
    This is a utility function needed to find unique nodes in the two subgraphs
    '''
    node1 = random.choice(list(G1_nodes['replier']))
    while node1 in list(G2_nodes['replier']):
        node1 = random.choice(list(G1_nodes['replier']))
    node2 = random.choice(list(G2_nodes['replier']))
    while node2 in list(G1_nodes['replier']):
        node2 = random.choice(list(G2_nodes['replier']))
        
    return node1, node2 


def functionality4(i1_start, i1_end, i2_start, i2_end, dataframe, user_1='700338', user_2='1522522'):
    '''
    Our functionality 4, this takes an interval of time and divides G
    into two subgraphs each containing uniquely user_1 and user2. 
    '''
    
    # this splits G into two subgraphs
    print("Extracting nodes")
    G1_nodes = dataframe[(dataframe['date'] > i1_start) &  (dataframe['date'] < i1_end)]
    G2_nodes = dataframe[(dataframe['date'] > i2_start) &  (dataframe['date'] < i2_end)]
    print("Merging dataframe")
    
    # here we merge it into one big df
    merged_df = pd.concat([G1_nodes, G2_nodes])
    # we then extract a graph
    G = nx.from_pandas_edgelist(merged_df,source='replier',target='questioner',edge_attr='weight', create_using=nx.DiGraph())
    
    # compute min cut
    print("Computing min cut")
    best_cut, weight = minCut(G, user_1, user_2)
    print("Runnning visualization")
    # run visualization
    visualization4(G, user_1, user_2, best_cut)
    
    return best_cut, weight, len(best_cut)
    
# visualization 4
def visualization4(G, user1, user2, cut_edges):
    '''
    Plots our feature 4!
    This function plots G and the edges requires to disconnect it in two subgraphs containing
    respectively user1, and user2
    
    Params:
        [G] - graph
        [user1] - unique subgraph1 user
        [user2] - unique subgraph2 user
        [cut_edges] - a list of edges involved in the min cut
    '''
    
    # defining a list of nodes that will be plotted 
    nodes_to_iter = set([user1, user2])
    
    # iterate through edges and their child and find nodes that will be plotted
    for edge in cut_edges:
        nodes_to_iter.update([edge[0], edge[1]])
    # every iteration add children nodes to the list
    for _ in range(3) :
        for node in list(nodes_to_iter) : 
            nodes_to_iter.update(list(G.neighbors(node))[:7])
    
    # plot the subgraph
    plt.figure(figsize=(15, 10), dpi=100)
    seed = 5587504
    G_2 = nx.subgraph(G, nodes_to_iter)
    pos2 = nx.spring_layout(G_2, seed = seed)
    nx.draw_networkx_nodes(G_2, pos2, node_size=200, nodelist=list(nodes_to_iter), 
                         node_color='white',)

    # draw first edges
    nx.draw_networkx_edges(G_2, pos2, width=1, alpha=0.6, edge_color = 'deepskyblue')
    
    # plot user1 and user2 nodes
    nx.draw_networkx_nodes(G_2, pos2, node_size=800, 
                         node_color='darkcyan', nodelist=[user1, user2])
    # plot and highlight cut_edges
    for edge in cut_edges :
        nx.draw_networkx_edges(G_2, pos2, width= 10, edgelist = [(edge[0], edge[1])], alpha = 0.8,
                               edge_color='Crimson', 
                                arrowsize=15, connectionstyle='arc3,rad=0.05')
    
    plt.title(f'MinCut edges to disconnect user {user1} and {user2}');
    plt.show()