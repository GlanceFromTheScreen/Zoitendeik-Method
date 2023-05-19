from Zoitendeik import *
from math import sqrt


def matrix_norma(m):
    res = 0
    for line in m:
        for it in line:
            res += it ** 2
    return sqrt(res)


def find_R(P):
    m = lambda x: [[9 / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + 27 * x[0]**2 / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                   9 * x[0] * x[1] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                   9 * x[0] * x[2] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2)],

                   [3 * x[1] * 3 * x[0] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                   3 / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + 3 * x[1]**2 / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                   3 * x[1] * 3 * x[2] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2)],

                   [3 * x[2] * 3 * x[0] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                   3 * x[2] * x[1] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                   3 / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + 3 * x[2]**2 / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2)]]

    n = 5
    t0 = [P[0][0] + (P[0][1] - P[0][0]) / n * i for i in range(n + 1)]
    t1 = [P[1][0] + (P[1][1] - P[1][0]) / n * i for i in range(n + 1)]
    t2 = [P[2][0] + (P[2][1] - P[2][0]) / n * i for i in range(n + 1)]

    points = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                points.append([t0[i], t1[j], t2[k]])

    R = max([matrix_norma(m(points[i])) for i in range(len(points))])

    return R


def find_K(ztd, P):  # P - прямоугольник в R^3: P = [[0, 1], [0, 1], [0,1]]
    n = 5
    t0 = [P[0][0] + (P[0][1] - P[0][0]) / n * i for i in range(n + 1)]
    t1 = [P[1][0] + (P[1][1] - P[1][0]) / n * i for i in range(n + 1)]
    t2 = [P[2][0] + (P[2][1] - P[2][0]) / n * i for i in range(n + 1)]

    points = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                points.append([t0[i], t1[j], t2[k]])

    K = []
    for ctr in ztd.phi_list:
        K.append(max([norma_calculate(ctr.grad(points[i])) for i in range(len(points))]))

    return max(K)





