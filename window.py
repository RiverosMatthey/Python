import numpy as np
from scipy.stats import zscore, kurtosis, skew

def f(n):
    C = n.T
    b = C
    A = zscore(b)
    A = A.T

    window_width = 150
    incres = 75
    pasos = (A.shape[0] - window_width) // incres

    medias = np.zeros((pasos, A.shape[1]))
    mediana = np.zeros((pasos, A.shape[1]))
    minimos = np.zeros((pasos, A.shape[1]))
    maximos = np.zeros((pasos, A.shape[1]))
    rango = np.zeros((pasos, A.shape[1]))
    raiz = np.zeros((pasos, A.shape[1]))
    Curtosis = np.zeros((pasos, A.shape[1]))
    skew = np.zeros((pasos, A.shape[1]))

    for i in range(pasos):
        medias[i, :] = np.mean(A[i:i+window_width, :], axis=0)
        mediana[i, :] = np.median(A[i:i+window_width, :], axis=0)
        minimos[i, :] = np.min(A[i:i+window_width, :], axis=0)
        maximos[i, :] = np.max(A[i:i+window_width, :], axis=0)
        rango[i, :] = np.ptp(A[i:i+window_width, :], axis=0)
        raiz[i, :] = np.sqrt(np.mean(np.square(A[i:i+window_width, :]), axis=0))
        Curtosis[i, :] = kurtosis(A[i:i+window_width, :], axis=0)
        #skew[i, :] = skewness(A[i:i+window_width, :], axis=0)

    matrizfinal = np.hstack((medias, mediana, minimos, maximos, rango, raiz, Curtosis, skew))
    return matrizfinal