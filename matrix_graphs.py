import math


def spaning_trees(M):
    return matrix_cofactor(M, 0, 0)

#create laplacian matrix for a simple graph
def create_laplacian(M):
    M2 = M
    for i in range(len(M)):
        deg = 0
        for j in range(len(M[i])):
            if M[i][j] > 0:
                deg += 1
        M2[i][i] = deg
    return M2


#calculate
def matrix_cofactor(M, i, j):
    if len(M) < 3:
        print("matrix to small")
        return 0
    if len(M) != len(M[0]):
        print("not a square matrix")
        return 0
    lap_m = create_laplacian(M)
    lap_m = det_sub_matrix(lap_m, i, j)
    return det(lap_m);


#prints the matrix
def print_matrix(A):
    for i in range(len(A)):
        print(A[i])


# TODO: add reduced ref calculations
# 
# recursivly calculate determinant
def det(M):
    if len(M) < 2:
        print("matrix to small")
        return 0
    if len(M) != len(M[0]):
        print("not a square matrix")
        return 0
    if len(M) == 2:
        return M[0][0] * M[1][1] - M[1][0] * M[0][1]
    else:
        dets = [0 for x in range(len(M))]
        for i in range(len(M)):
            dets[i] = det(det_sub_matrix(M,0,i))
        #add and subtract determinants
        determ = 0
        for i in range(len(M)):
            tmp = dets[i] * M[0][i]
            if i % 2 == 1:
                tmp = -tmp
            determ += tmp
    return determ

def det_sub_matrix(M, indexI, indexJ):
    M2 = [[0 for x in range(len(M) - 1)] for x in range(len(M) - 1)]
    for i in range(len(M)): # horizontal
        if i != indexI:
            for j in range(len(M)):
                if j != indexJ:
                    i_pos = i
                    j_pos = j
                    if i > indexI:
                        i_pos = i - 1
                    if j > indexJ:
                        j_pos = j - 1
                    M2[i_pos][j_pos] = M[i][j]
    return M2

#degree sequence
def deg_seq(M):
    d = []
    for i in range(len(M)):
        deg = 0
        for j in range(len(M)):
            if M[i][j] != 0:
                deg += 1
        d.append(deg)
    return d

#is eularian
#input takes a degree sequence
#returns true of eularian or semi eularian
#todo change from true false to three
def is_eularian(d):
    odd = 0
    for i in range(len(d)):
        if 0 != d[i] % 2:
            odd += 1
            if odd > 2:
                return False
    if odd == 1:
        return False
    return True

#reverse directions
def reverse_graph(M):
    print_matrix(M)
    for i in range(len(M)):
        for j in range(len(M)):
            if i > j:
                print("i:", i)
                print("j:", j)
                temp = M[i][j]
                M[i][j] = M[j][i]
                M[j][i] = temp
    print_matrix(M)
    return M

#copys matrix to a new
def copy_matrix(M):
    M2 = []
    for i in range(len(M)):
        M2.append([])
        for j in range(len(M)):
            M2[i].append(M[i][j])
    return M2
#-------------------------------------------------------------------------------
#dijkstras works for both simple a digraphs
def dijkstra(M, origin):
    #dict of edges of tuples, (dest, origin, distance)
    tree = {}
    #add origin to tree
    tree[origin] = (origin, 0)
    #priority queue for distances
    Q = {}
    
    for i in range(len(M)):
        if i != origin:
            insert_queue(Q, M[origin][i], origin, i)
    
    #add smallest element to tree
    smallest = smallest_queue(Q)
    tree[smallest[0]] = smallest[1]
    last_add = smallest[0]
    
    #till the tree list is not full
    while len(tree) < len(M):
        #add edges from the newly added vertex
        add_new_vertex(Q, M, tree, last_add)
        #add smallest element to tree
        smallest = (smallest_queue(Q))
        tree[smallest[0]] = smallest[1]
        last_add = smallest[0]
    return tree

def insert_queue(Q, d, origin, dest):
    if not (dest in Q):
        Q[dest] = (origin, d)
    else:
        if d < Q[dest][1]:
            Q[dest] = (origin, d)

def smallest_queue(Q):
    min_key = -1
    min_dist = math.inf
    for k in Q:
        if Q[k][1] <= min_dist:
            min_key = k
            min_dist = Q[k][1]
    if min_key == -1:
        print("error")
    return (min_key, Q.pop(min_key))

def add_new_vertex(Q, M, tree, vertex):
    for i in range(len(M)):
        if i != vertex and not (i in tree):
            #add elements useing distance from vertex to i and distance from origin to i
            insert_queue(Q, M[vertex][i] + tree[vertex][1], vertex, i)
    
#-------------------------------------------------------------------------------

#kruscal
# input: simple matrix
# returns minimum spaning tree
def kruscal(M):
    M2 = [[0 for i in range(len(M))] for i in range(len(M))]
    #number of edges added
    edges = 0
    #list for what sub tree each vertex is in
    V = [i for i in range(len(M))]
    #get edges
    E = get_edges(M)
    #sort edges
    E.sort
    #get edges
    while edges < len(M) -1:
        w, v1, v2 = E.pop(0)
        #check if in same subtree
        if V[v1] != V[v2]:
            update_vertex(V, v1, v2)
            M2[v1][v2] = w
            M2[v2][v1] = w
            edges += 1
    return M2

def get_edges(M):
    E = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            if j > i:
                if M[i][j] != math.inf:
                    E.append((M[i][j], i, j))
    return E

def update_vertex(V, v1, v2):
    sub = V[v2]
    for i in range(len(V)):
        if V[i] == sub:
            V[i] = V[v1]
#-------------------------------------------------------------------------------


#floyd eularian
#deal withonly going to proper verticies
def floyd_eularian(M):
    M2 = copy_matrix(M)
    d = deg_seq(M2)
    d_sum = sum(d) / 2
    
    path = []
    
    cur = get_floyd_start(d)
    #do stuff
    while d_sum > 0:
        path.append(get_edge(M2, d, cur))
        cur = path[-1][1]
        d_sum -= 1
    return path

def get_floyd_start(d):
    for i in range(len(d)):
        if d[i] % 2 == 1:
            return i
    return 0

#select proper verticies
def get_edge(M, d, cur):
    for i in range(len(M)):
        if i != cur:
            if M[cur][i] != 0:
                if d[cur] == 1 or d[i] > 1:
                    d[cur] -= 1
                    d[i] -= 1
                    #check if want it
                    M[cur][i] = 0
                    M[i][cur] = 0
                    return (cur, i)
    print("error")
    return
#-------------------------------------------------------------------------------
# test help functions

def test_helper():
    failed = 0
    print("Testing helper functions")
    failed += test_deg_seq()
    
    if failed > 0:
        print("Failed", failed, "tests.")
    else:
        print("All tests passed.")
    return

def test_deg_seq():
    failed = 0
    print("Testing deg_seq function")
    
    #test deg_seq with simle graph
    print("deg_seq simple graph")
    simple_graph = [[0,1,0,1,1],[1,0,0,1,0],[0,0,0,1,1],[1,1,1,0,0],[1,0,1,0,0]]
    simple_deg_seq = [3,2,2,3,2]
    
    deg_seq1 = deg_seq(simple_graph)
    #test if correct
    if deg_seq1.sort() == simple_deg_seq.sort():
        print("passed")
    else:
        print("failed")
        failed += 1
    
    #test deg_seq with weighted graph
    print("deg_seq weighted graph")
    weighted_graph = [[0,3,4,0,1],[3,0,7,0,12],[4,7,0,3,0],[0,0,3,0,5],[1,12,0,5,0]]
    weighted_deg_seq = [3,3,3,2,3]
    
    deg_seq2 = deg_seq(weighted_graph)
    #test if correct
    if deg_seq2.sort() == weighted_deg_seq.sort():
        print("passed")
    else:
        print("failed")
        failed += 1
    
    return failed

#testing stuff
inf = math.inf
s_test = [[1,2],
          [3,4]]

test = [[0,1,1,1],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,0]]

t4 = [[j*4+i+1 for i in range(4)] for j in range(4)]

def gen_matrix(size):
    return [[j*size+i+1 for i in range(size)] for j in range(size)]

d_test = [[  0,  7,  9, 14,inf,inf],
          [  7,  0, 10, 15,inf,inf],
          [  9, 10,  0, 11,inf,  2],
          [ 14, 15, 11,  0,  6,inf],
          [inf,inf,inf,  6,  0,  9],
          [inf,inf,  2,inf,  9,  0]]

e_test = [[0, 1, 1, 0, 1, 1],
          [1, 0, 1, 0, 0, 0],
          [1, 1, 0, 1, 0, 1],
          [0, 0, 1, 0, 1, 0],
          [1, 0, 0, 1, 0, 0],
          [1, 0, 1, 0, 0, 0]]