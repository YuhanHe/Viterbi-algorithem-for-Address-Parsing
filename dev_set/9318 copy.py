import numpy as np
from collections import defaultdict
import copy

def string_token(str):
    L = []
    s = str.split()
    j = 0
    while True:
        if j == len(s):
            break
        i = 0
        k = 0
        while True:
            if i == len(s[j]):
                L.append(s[j][k:i])
                break
            if s[j][i] == '*' or s[j][i] == ',' or s[j][i] == '(' or s[j][i] == ')' or s[j][i] == '/' or s[j][
                i] == '-' or s[j][i] == '&' or s[j][i] == '*':
                L.append(s[j][k:i])
                L.append(s[j][i])
                k = i + 1
            i += 1
        j += 1

    final_L = []
    for ele in L:
        if ele != '':
            final_L.append(ele)
    return final_L


def read_state_file(file_name):
    file = open(file_name, 'r')
    L = []
    message = file.readlines()
    for ele in message:
        L.append(ele.split('\n'))
    L_new = []
    for ele in L:
        L_new.append(ele[0].strip())
    number_of_state = int(L_new[0])
    state_list = []
    for i in range(number_of_state + 1):
        state_list.append(L_new[i])
    N_ij = np.zeros((number_of_state, number_of_state), dtype=np.int)
    N_i = [0 for i in range(number_of_state)]
    for k in range(number_of_state + 1, len(L_new)):
        ele = L_new[k].split()
        N_ij[int(ele[0])][int(ele[1])] = int(ele[2])
        N_i[int(ele[0])] += int(ele[2])
    A_ij = np.zeros((number_of_state, number_of_state), dtype=np.float)
    for i in range(number_of_state):
        for j in range(number_of_state):
            A_ij[i][j] = float((N_ij[i][j] + 1) / (number_of_state + N_i[i] - 1))
    for i in range(number_of_state):
        A_ij[i][-2] = 0
        A_ij[-1][i] = 0
    return number_of_state, A_ij


def read_symbol_file(state_file_name, symbol_file_name):
    number_of_state, A_ij = read_state_file(state_file_name)
    file = open(symbol_file_name, 'r')
    L = []
    message = file.readlines()
    for ele in message:
        L.append(ele.split('\n'))
    L_new = []
    for ele in L:
        L_new.append(ele[0].strip())
    number_of_symbol = int(L_new[0])
    symbol_dic = defaultdict()
    j = 0
    for i in range(1, number_of_symbol + 1):
        symbol_dic[L_new[i]] = j
        j += 1
    B_ij = np.zeros((number_of_state, number_of_symbol + 1), dtype=np.float)
    N_ij = np.zeros((number_of_state, number_of_symbol + 1), dtype=np.int)
    N_i = [0 for i in range(number_of_state)]
    for k in range(number_of_symbol + 1, len(L_new)):
        ele = L_new[k].split()
        N_ij[int(ele[0])][int(ele[1])] = int(ele[2])
        N_i[int(ele[0])] += int(ele[2])
    for i in range(number_of_state):
        for j in range(number_of_symbol):
            B_ij[i][j] = float((N_ij[i][j] + 1) / (number_of_symbol + N_i[i] + 1))
    for i in range(number_of_state):
        B_ij[i][number_of_symbol] = float(1 / (number_of_symbol + N_i[i] + 1))
    for i in range(number_of_state - 2, number_of_state):
        for j in range(number_of_symbol + 1):
            B_ij[i][j] = 0
    return symbol_dic, B_ij, number_of_symbol


def open_query_file(file_name):
    file = open(file_name, 'r')
    L = []
    message = file.readlines()
    for ele in message:
        L.append(ele.split('\n'))
    L_new = []
    for ele in L:
        L_new.append(ele[0].strip())
    add_list = []
    for ele in L_new:
        e = string_token(ele)
        add_list.append(e)
    return add_list


def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    N, A = read_state_file(State_File)
    # print(N)
    # print(A)
    symbol_dic, B, M = read_symbol_file(State_File, Symbol_File)
    # print(M)
    # print(symbol_dic)
    # print(B)
    queries = open_query_file(Query_File)
    # print(queries)
    results = []
    for query in queries:
        symbols = []
        for i in range(len(query)):
            if query[i] in symbol_dic:
                symbols.append(symbol_dic[query[i]])
            else:
                symbols.append(M)
        # print(symbols)
        results.append(query_handle(symbols, N, A, M, B))
    return results

def query_handle(symbols, N, A, M, B):
    print('ss')

    # N is the number of state, A is A_ij, M is the number of symbol, B is B_ij
    states = []
    P = np.zeros((N-2, len(symbols)), dtype=np.float)      # Store prob.
    Q = []                                                 # Store the path

    for i in range(N-2):                                   # symbol is the symbol index
        Q.append([])
        for j in range(len(symbols)):
            Q[i].append([])
            if j == 0:
                Q[i][j].append(N - 2)
                Q[i][j].append(i)

    for i in range(N-2):
        P[i][0] = np.log(A[N - 2][i]) +np.log(B[i][symbols[0]])
    print(P)

    for j in range(1, len(symbols)):
        for i in range(N-2):
            max_prob = 0
            index = 0
            for k in range(N-2):
                a = A[k][i]
                b = B[i][symbols[j]]
                p = np.log(a) + np.log(b) + P[k][j-1]
                # print(p)
                if k == 0:
                    max_prob = p
                if p >= max_prob:
                    index = k
                    max_prob = p

            P[i][j] = max_prob
            L = copy.deepcopy(Q[index][j-1])
            L.append(i)
            Q[i][j] = L

    max_value = 0
    index_track = 0
    for i in range(N-2):
        value = np.log(A[i][-1]) + P[i][-1]
        if i == 0:
            max_value = value
        if max_value <= value:
            index_track = i
            max_value = value

    states.extend(Q[index_track][-1])
    states.append(N-1)
    states.append(max_value)

    return states


def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    N, A = read_state_file(State_File)
    symbol_dic, B, M = read_symbol_file(State_File, Symbol_File)
    queries = open_query_file(Query_File)
    results = []
    for query in queries:
        symbols = []
        for i in range(len(query)):
            if query[i] in symbol_dic:
                symbols.append(symbol_dic[query[i]])
            else:
                symbols.append(M)
        results.append(query_handle_k_tops(symbols, N, A, M, B, k))
    return results


def query_handle_k_tops(symbols, N, A, M, B, k):
    Q = defaultdict(list)
    print('ss')
    states = []
    if k <= N-2:
        P = np.zeros((N - 2, len(symbols), k), dtype=np.float)
        p_1 = np.zeros((N - 2, len(symbols)), dtype=np.float)
        Q = []
        for i in range(N - 2):
            p_1[i][0] = np.log(A[N - 2][i]) + np.log(B[i][symbols[0]])
        for i in range(N-2):
            temp = []
            for h in range(N-2):
                a = A[h][i]
                b = B[i][symbols[1]]
                p = np.log(a) + np.log(b) + p_1[h][0]
                temp.append(p)
            temp.sort(reverse=True)
            for o in range(k):
                P[i][1][o] = temp[o]

        for j in range(2, len(symbols)):
            for i in range(N-2):
                temp = []
                for h in range(N-2):
                    a = A[h][i]
                    b = B[i][symbols[j]]
                    for w in range(k):
                        p = np.log(a) + np.log(b) + P[h][j - 1][w]
                        temp.append(p)

                temp.sort(reverse=True)
                for o in range(k):
                    P[i][j][o] = temp[o]

        final_list = []
        for i in range(N-2):
            for j in range(k):
                final_list.append(P[i][-1][j] + np.log(A[i][-1]))
        final_list.sort(reverse=True)
        # print(final_list)
        print_list = []
        for i in range(k):
            print_list.append(final_list[i])
        print(print_list)

        return states

    else:
        matrix = []
        for i in range(N-2):
            matrix.append([])
            for j in range(len(symbols)):
                matrix[i].append([])

        for i in range(N - 2):
            l = []
            l.append(np.log(A[N - 2][i]) + np.log(B[i][symbols[0]]))
            matrix[i][0] = l

        flag = True
        index = 0
        for j in range(1, len(symbols)):
            for i in range(N-2):
                for h in range(N-2):
                    if flag:
                        a = A[h][i]
                        b = B[i][symbols[j]]
                        for w in range(len(matrix[h][j-1])):
                            p = np.log(a) + np.log(b) + matrix[h][j - 1][w]
                            matrix[i][j].append(p)

                if len(matrix[i][j]) >= k:
                    matrix[i][j].sort(reverse=True)

            if len(matrix[i][j]) >= k:
                flag = False
                index = j

        for j in range(index + 1, len(symbols)):
            for i in range(N-2):
                temp = []
                for h in range(N-2):
                    a = A[h][i]
                    b = B[i][symbols[j]]
                    for w in range(k):
                        p = np.log(a) + np.log(b) + matrix[h][j - 1][w]
                        temp.append(p)
                temp.sort(reverse=True)
                l = []
                for ele in range(k):
                    l.append(temp[ele])
                matrix[i][j] = l

        final_list = []

        for i in range(N - 2):
            for j in range(len(matrix[0][-1])):
                final_list.append(matrix[i][-1][j] + np.log(A[i][-1]))
        final_list.sort(reverse=True)
        print_list = []
        for i in range(k):
            print_list.append(final_list[i])
        print(print_list)

# r = viterbi_algorithm('State_File_1', 'Symbol_File_1', 'Query_File_1')
# for ele in r:
#     print(ele)

k = top_k_viterbi('State_File_1', 'Symbol_File_1', 'Query_File_1', 2)
print(k)
