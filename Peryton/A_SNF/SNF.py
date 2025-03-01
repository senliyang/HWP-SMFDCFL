import numpy as np
import pandas as pd


# W is the matrix which needs to be normalized
def new_normalization(w):
    m = w.shape[0]
    p = np.zeros([m, m])

    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1 / 2
            elif np.sum(w[i, :]) - w[i, i] > 0:
                p[i][j] = w[i, j] / (2 * (np.sum(w[i, :]) - w[i, i]))
    return p


# Get the KNN kernel, k is the number of first nearest neighbors
def KNN_kernel(S, k):
    n = S.shape[0]
    S_knn = np.zeros([n, n])
    for i in range(n):
        sort_index = np.argsort(S[i, :])
        for j in sort_index[n - k:n]:
            if np.sum(S[i, sort_index[n - k:n]]) > 0:
                S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n - k:n]]))
    return S_knn


# Updating rules for microbe similarity
def MiRNA_updating(S1, S2, S3, P1, P2, P3):
    it = 0
    P = (P1 + P2 + P3) / 3
    dif = 1
    while dif > 0.0000001:
        it += 1
        P111 = np.dot(np.dot(S1, (P2 + P3) / 2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P3) / 2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3, (P1 + P2) / 2), S3.T)
        P333 = new_normalization(P333)

        P1 = P111
        P2 = P222
        P3 = P333

        P_New = (P1 + P2 + P3) / 3
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iteration (microbe):", it)
    return P


# Updating rules for disease similarity
def disease_updating(S1, S2, S3, P1, P2, P3):
    it = 0
    P = (P1 + P2 + P3) / 3
    dif = 1
    while dif > 0.0000001:
        it += 1
        P111 = np.dot(np.dot(S1, (P2 + P3) / 2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P3) / 2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3, (P1 + P2) / 2), S3.T)
        P333 = new_normalization(P333)

        P1 = P111
        P2 = P222
        P3 = P333

        P_New = (P1 + P2 + P3) / 3
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iteration (disease):", it)
    return P


# Main function to calculate synthetic similarity matrices
def get_syn_sim(k1, k2):
    # Load disease similarity matrices
    disease_sim1 = np.loadtxt('DSS - 副本.txt', delimiter='\t')
    disease_sim2 = np.loadtxt("DC.txt", delimiter='\t')
    disease_sim3 = np.loadtxt("DG.txt", delimiter='\t')

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(disease_sim3)
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim2, k2)
    Sd_3 = KNN_kernel(disease_sim3, k2)

    Pd = disease_updating(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2


    # Load microbe similarity matrices
    microbe_sim1 = np.loadtxt('MC.txt', delimiter='\t')
    microbe_sim2 = np.loadtxt("MF.txt", delimiter='\t')
    microbe_sim3 = np.loadtxt("MG.txt", delimiter='\t')

    m1 = new_normalization(microbe_sim1)
    m2 = new_normalization(microbe_sim2)
    m3 = new_normalization(microbe_sim3)
    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(microbe_sim2, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)

    Pm = MiRNA_updating(Sm_1, Sm_2, Sm_3, m1, m2, m3)
    Pm_final = (Pm + Pm.T) / 2



    return Pm_final, Pd_final


if __name__ == '__main__':
    k1 = 140  # Number of nearest neighbors for microbe similarity
    k2 = 4  # Number of nearest neighbors for disease similarity
    sim_m, sim_d = get_syn_sim(k1, k2)
    np.savetxt('m_sim_final.txt', sim_m)
    np.savetxt('d_sim_final.txt', sim_d)
