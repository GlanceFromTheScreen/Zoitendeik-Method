from Zoitendeik import *
from math import sqrt

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

    z = Zoitendeik_step(phi0, [phi1_border_constraint, phi2_border_constraint, phi3, phi4], [0.0, 0.0, 0.0], 0.5, 0.5)

    z.minimize()
