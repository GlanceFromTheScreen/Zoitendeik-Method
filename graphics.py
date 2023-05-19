import matplotlib.pyplot as plt
from Zoitendeik import *
from math import sqrt
from constants_evaluate import find_R, find_K


def graph(ztd, f_star):
    eps = 0.01
    arr = []

    ztd.I_upd()
    ztd.x = ztd.find_x0()
    i = 0
    while True:  # поставить нормальное условие
        ztd.I_upd()
        ztd.s_upd()
        ztd.lmd_upd()
        ztd.x_upd()

        arr.append(abs(ztd.f0.f(ztd.x) - f_star))

        if ztd.eta == 0 and ztd.dlt < eps and ztd.dlt < -1 * max([ztd.phi_list[j].phi(ztd.x) for j in ztd.Id]):
            break

        if -ztd.eta < eps and ztd.dlt < eps:
            break

        i += 1
        if i == 200:
            break

    print(arr)
    plt.plot(range(len(arr)), arr, '-o')
    plt.title('Решение на границе')
    plt.xlabel('iteration')
    plt.ylabel('|f(x_k) - f*|')
    plt.semilogy()
    plt.show()


if __name__ == '__main__':
    #  [-0.12133422914447144, -0.3542620827004164, -0.18271647615185854] - absolute min
    phi0 = Target_function(lambda x: x[0] + x[1] + 0.5 * x[2] + 3 * sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                           lambda x: [1 + 9 * x[0] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                                      1 + 3 * x[1] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
                                      0.5 + 3 * x[2] / sqrt(1 + 3 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2)])

    phi1 = Constraint('ineq',
                      lambda x: x[0] ** 2 + x[1] ** 2 - 1,
                      lambda x: [2 * x[0], 2 * x[1], 0])

    phi1_border_constraint = Constraint('ineq',
                                        lambda x: (x[0] - 1) ** 2 + (x[1] - 1) ** 2 - 1,
                                        lambda x: [2 * (x[0] - 1), 2 * (x[1] - 1), 0])

    phi2 = Constraint('ineq',
                      lambda x: x[0] ** 2 + x[2] ** 2 - 1,
                      lambda x: [2 * x[0], 0, 2 * x[2]])

    phi2_border_constraint = Constraint('ineq',
                                        lambda x: (x[0] - 1) ** 2 + (x[2] - 1) ** 2 - 1,
                                        lambda x: [2 * (x[0] - 1), 0, 2 * (x[2] - 1)])

    phi3 = Constraint('ineq',
                      lambda x: x[1] ** 2 + x[2] ** 2 - 1,
                      lambda x: [0, 2 * x[1], 2 * x[2]])

    phi4 = Constraint('eq',
                      lambda x: x[1] - (0.35426 / 0.121334) * x[0],
                      lambda x: [- (0.35426 / 0.121334), 1, 0])

    z = Zoitendeik_step(phi0, [phi1_border_constraint, phi2_border_constraint, phi3, phi4], [0.0, 0.0, 0.0], 0.25, 0.5)

    k = find_K(z, [[0, 1], [0, 1], [0, 1]])
    r = find_R([[0, 1], [0, 1], [0, 1]])  # не применять find_R для других условий!!!
    print('K: ', k)
    print('R: ', r)

    z.f0.R = r
    for ctr in z.phi_list:
        ctr.K = k
        ctr.R = r

    f_star = 4.522737127133121
    graph(z, f_star)

