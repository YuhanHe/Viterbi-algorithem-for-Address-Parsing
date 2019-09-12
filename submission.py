import numpy as np
from collections import defaultdict
import copy
import re
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
            if s[j][i] == ',' or s[j][i] == '(' or s[j][i] == ')' or s[j][i] == '/' or s[j][
                i] == '-' or s[j][i] == '&':
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
    state_dic = defaultdict()
    state_list = []
    j = 0
    for i in range(number_of_state + 1):
        state_list.append(L_new[i])
        if i != 0:
            state_dic[L_new[i]] = j
            j+=1
    N_ij = np.zeros((number_of_state, number_of_state), dtype = np.int)
    N_i = [0 for i in range(number_of_state)]
    for k in range(number_of_state + 1, len(L_new)):
        ele = L_new[k].split()
        N_ij[int(ele[0])][int(ele[1])] = int(ele[2])
        N_i[int(ele[0])] += int(ele[2])
    A_ij = np.zeros((number_of_state, number_of_state), dtype = np.float)
    for i in range(number_of_state):
        for j in range(number_of_state):
            A_ij[i][j] = float((N_ij[i][j] + 1) / (number_of_state + N_i[i] - 1))
    for i in range(number_of_state):
        A_ij[i][-2] = 0
        A_ij[-1][i] = 0
    return number_of_state, A_ij, state_dic

def read_symbol_file2(state_file_name, symbol_file_name):
    number_of_state, A_ij , state_dic = read_state_file(state_file_name)
    file = open(symbol_file_name, 'r')
    L = []
    message = file.readlines()
    for ele in message:
        L.append(ele.split('\n'))
    L_new = []
    for ele in L:
        L_new.append(ele[0].strip())
    number_of_symbol = int(L_new[0])
    N = number_of_symbol
    symbol_dic = defaultdict()
    j = 0
    for i in range(1, number_of_symbol + 1):
        symbol_dic[L_new[i]] = j
        j += 1
    B_ij = np.zeros((number_of_state, number_of_symbol + 1), dtype = np.float)
    C = np.zeros((number_of_state, number_of_symbol + 1), dtype = np.int)
    T = [0 for i in range(number_of_state)]
    V = [0 for i in range(number_of_state)]
    for k in range(number_of_symbol + 1, len(L_new)):
        ele = L_new[k].split()
        C[int(ele[0])][int(ele[1])] = int(ele[2])
        T[int(ele[0])] += int(ele[2])
        V[int(ele[0])] += 1
    for i in range(number_of_state-2):
        for j in range(number_of_symbol+1):
            P = 1/(T[i] + V[i])
            if C[i][j] == 0:
                B_ij[i][j] = V[i]*P/(N-V[i]+1)
            else:
                B_ij[i][j] = C[i][j]/T[i] - P
    return symbol_dic, B_ij, number_of_symbol

def read_symbol_file(state_file_name, symbol_file_name):
    number_of_state, A_ij , state_dic = read_state_file(state_file_name)
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

def query_handle_k_tops(symbols, N, A, M, B, k):
    matrix = [] # probobility matrix

    for i in range(N-2):
        matrix.append([])
        for j in range(len(symbols)):
        #for j in range(k):
            matrix[i].append([])
    # init the first column of matrix
    for i in range(N - 2):
        l = []
        l.append(np.log(A[N - 2][i]) + np.log(B[i][symbols[0]]))
        matrix[i][0] = l

    Q = []      # Q for storing state sequences
    for i in range(N - 2):  # initialize the state matrix Q
        Q.append([])
        for j in range(len(symbols)):
            Q[i].append([])
            if j == 0:
                temp_l = []
                temp_l.append(N - 2)
                temp_l.append(i)
                Q[i][j].append((matrix[i][j][0],temp_l))

    flag = True
    index = 0
    # start from second symbol
    for j in range(1, len(symbols)):
        if flag == False:
            break
        for i in range(N-2): # i is current state ID
            for h in range(N-2): # h is previous state ID
                if flag: # less than k probs
                    a = A[h][i] # transition probobility from state h to state i
                    b = B[i][symbols[j]] # emission probobility from state i to current symbol

                    for w in range(len(matrix[h][j-1])):
                        p = np.log(a) + np.log(b) + matrix[h][j - 1][w]
                        matrix[i][j].append(p)
                        temp_3 = copy.deepcopy(Q[h][j - 1][w][1])
                        temp_3.append(i)
                        Q[i][j].append((p,temp_3))

            if len(matrix[i][j]) >= k: # for selection of top_k prob
                matrix[i][j].sort(reverse=True)
                temp_2 = copy.deepcopy(matrix[i][j][:k])
                matrix[i][j] = temp_2

                Q[i][j].sort(key = lambda x:(-x[0], x[1]))
                temp_6 = copy.deepcopy(Q[i][j][:k])
                Q[i][j] = temp_6

        if len(matrix[i][j]) >= k:
            flag = False # more than k probs
            index = j

    if flag == True:
        final_list = []
        for i in range(N - 2):
            for j in range(len(matrix[i][-1])):
                p = matrix[i][-1][j] + np.log(A[i][-1])
                temp_5 = copy.deepcopy(Q[i][-1][j][1])
                temp_5.append(N-1)
                final_list.append((p,temp_5[::-1]))
                
        final_list.sort(key=lambda x: (-x[0], x[1]))
        print_list = []
        for i in range(k):
            result = []
            result = copy.deepcopy(final_list[i][1][::-1])
            result.append(final_list[i][0])
            print_list.append(result)
        return print_list        
    else:    
        for j in range(index + 1, len(symbols)):
            for i in range(N-2):
                temp = []
                for h in range(N-2):
                    a = A[h][i]
                    b = B[i][symbols[j]]
                    for w in range(k):
                        p = np.log(a) + np.log(b) + matrix[h][j - 1][w]
                        temp_4 = copy.deepcopy(Q[h][j - 1][w][1])
                        temp_4.append(i)
                        temp.append((p,temp_4[::-1]))
                temp.sort(key=lambda x: (-x[0], x[1]))
                l = []
                l2 = []

                for ele in range(k):
                    l.append(temp[ele][0])
                    l2.append((p,temp[ele][1][::-1]))
                matrix[i][j] = l
                Q[i][j] = copy.deepcopy(l2)

        final_list = []

        for i in range(N - 2):
            for j in range(len(matrix[i][-1])):
                p = matrix[i][-1][j] + np.log(A[i][-1])
                temp_5 = copy.deepcopy(Q[i][-1][j][1])
                temp_5.append(N-1)
                final_list.append((p,temp_5[::-1]))
   
        final_list.sort(key=lambda x: (-x[0], x[1]))
        print_list = []
        for i in range(k):
            result = []
            result = copy.deepcopy(final_list[i][1][::-1])
            result.append(final_list[i][0])
            print_list.append(result)
        return print_list

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    return top_k_viterbi(State_File, Symbol_File, Query_File, 1)

def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    N, A , state_dic = read_state_file(State_File)
    symbol_dic, B, M = read_symbol_file(State_File, Symbol_File)
    queries = open_query_file(Query_File)
    results = []
    for query in queries:
        symbols = []
        for i in range(len(query)):
            if query[i] in symbol_dic: # the symbol is not UNK
                symbols.append(symbol_dic[query[i]]) # the symbolID for known symbols
            else: # the symbol is UNK
                symbols.append(M) # the symbolID for UNK is M
        result = query_handle_k_tops(symbols, N, A, M, B, k)
        for i in range(len(result)):
            results.append(result[i])
    return results
        
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    N, A , state_dic = read_state_file(State_File)
    symbol_dic, B, M = read_symbol_file2(State_File, Symbol_File)
    queries = open_query_file(Query_File)
    results = []
    for query in queries:
        symbols = []
        for i in range(len(query)):
            if query[i] in symbol_dic: # the symbol is not UNK
                symbols.append(symbol_dic[query[i]]) # the symbolID for known symbols
            else: # the symbol is UNK
                symbols.append(M) # the symbolID for UNK is M
        result = query_handle_k_tops(symbols, N, A, M, B, 1)
        for i in range(len(result)):
            results.append(result[i])
    return results




