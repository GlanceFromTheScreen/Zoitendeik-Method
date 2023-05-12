import numpy as np
from simplex_lib.preprocessing import Make_Canon_Form, Update_C
from simplex_lib.simplex import Simplex_With_Init, Recover_Initial_Variables


class Target_function:
    def __init__(self, f_, grad_):
        self.f = f_
        self.grad = grad_
        self.R = None
        self.K = None


class Constraint:
    def __init__(self, type_, phi_, grad_):
        self.type = type_  # eq / ineq
        self.phi = phi_
        self.grad = grad_
        self.R = None
        self.K = None


class Zoitendeik_step:
    def __init__(self, f_, phi_list, x0, delta0, lambda0):
        self.f0 = f_
        self.phi_list = phi_list  # list of constraints
        self.x = x0
        self.dlt = delta0
        self.lmd = lambda0
        self.eta = None
        self.s = []
        self.I0 = []
        self.I1 = []
        self.Id = []  # I_delta

    def I_upd(self):
        i = 0  # здесь нумерация не с 1, как у Е.А., а с 0
        for constraint in self.phi_list:
            phi_value = constraint.phi(self.x)
            phi_type = constraint.type
            if phi_type == 'ineq':
                if phi_value == 0:
                    self.I0.append(i)
                    self.Id.append(i)
                elif phi_value < -self.dlt:
                    self.I1.append(i)
                else:
                    self.Id.append(i)
            i += 1

    def make_matrix(self):
        """
        Создаем матрицу ограничений и вектор свободных
        членов для применения симплекс-метода в методе s_upd"""
        matrix = []
        b = []

        line0 = self.f0.grad(self.x)  # градиент целевой функции в точке 'x'
        line0.append(-1.0)  # добавили коэффициент -1 для eta
        matrix.append(line0)
        b.append(0.0)
        for i in self.Id:  # первые сторчки - ограничения - неравентсва (почти актиыные) (такой порядок нкжен для simplex-а)
            line = self.phi_list[i].grad(self.x)
            line.append(-1.0)
            matrix.append(line.copy())
            b.append(0.0)

        for i in range(len(line0) - 1):  # eta не учитываем (записываем условия на норму направляющего вектора)
            line = [0.0 for j in line0]
            line[i] = 1.0
            matrix.append(line.copy())
            b.append(1.0)

            line[i] = -1.0
            matrix.append(line.copy())
            b.append(1.0)

        # for i in self.I0:  # последние строчки - линейные ограничения - равентсва
        #     line = self.phi_list[i].grad(self.x)
        #     line.append(-1)
        #     matrix.append(line.copy())
        #     b.append(0)

        c = [0.0 for i in line0]
        c[-1] = 1.0

        eq_less = 1 + len(self.Id) + 2*(len(line0) - 1)  # phi0, phi_i, -1 < s_j < 1

        return matrix, b, c, eq_less

    def s_upd(self):
        """
        Находим направление спуска
        """
        M, b, c, eq_less = self.make_matrix()
        x_any = len(c)
        A2, b2, Ind2 = Make_Canon_Form(np.array(M).copy(),
                                       np.array(b),
                                       False,
                                       0,
                                       eq_less,
                                       0)

        c2, c_free2 = Update_C(A2.copy(), b2.copy(), np.array(c).copy(), 0, Ind2, eq_less, 0, x_any)

        opt_vector, self.eta = Simplex_With_Init(A2.copy(), b2.copy(), Ind2.copy(), c2.copy(), c_free2.copy())
        self.s = Recover_Initial_Variables(opt_vector, x_any)

    def minimize(self):
        self.I_upd()


if __name__ == '__main__':

    phi0 = Target_function(lambda x: 2*x[0]**2 + 2*x[1]**2 - 2*x[0]*x[1] - 4*x[0] - 6*x[1],
                           lambda x: [4*x[0] - 2*x[1] - 4, 4*x[1] - 2*x[0] - 6])

    phi1 = Constraint('ineq',
                      lambda x: x[0] + 5*x[1] - 5,
                      lambda x: [1, 5])

    phi2 = Constraint('ineq',
                      lambda x: 2*x[0]**2 - x[1],
                      lambda x: [4*x[0], -1])

    phi3 = Constraint('ineq',
                      lambda x: -x[0],
                      lambda x: [-1, 0])

    phi4 = Constraint('ineq',
                      lambda x: -x[1],
                      lambda x: [0, -1])

    z = Zoitendeik_step(phi0, [phi1, phi2, phi3, phi4], [0.0, 0.75], 0.0001, 0.4)

    z.I_upd()
    z.make_matrix()
    z.s_upd()

    print('cat')





