import numpy as np
from simplex_lib.preprocessing import Make_Canon_Form, Update_C
from simplex_lib.simplex import Simplex_With_Init, Recover_Initial_Variables
from itertools import combinations

np.seterr(divide='ignore', invalid='ignore')


def norma_calculate(v):
    res = 0
    for item in v:
        res += item ** 2
    return res ** (1 / 2)


class Target_function:
    def __init__(self, f_, grad_, R=10):
        self.f = f_
        self.grad = grad_
        self.R = R  # предполагается, что потом надо изменить


class Constraint:
    def __init__(self, type_, phi_, grad_, R=10, K=10):
        self.type = type_  # eq / ineq
        self.phi = phi_
        self.grad = grad_
        self.R = R  # предполагается, что потом надо изменить
        self.K = K


class Zoitendeik_step:
    def __init__(self, f_, phi_list, x0, delta0, alfa):
        self.f0 = f_
        self.phi_list = phi_list  # list of constraints
        self.x = x0
        self.dlt = delta0
        self.lmd = None  # step size
        self.alfa = alfa  # параметр дробления (не шаг)
        self.eta = None
        self.s = []
        self.I0 = []
        self.I1 = []
        self.Id = []  # I_delta
        self.eq = []  # equations indexes among phi_list

    def I_upd(self):
        i = 0  # здесь нумерация не с 1, как у Е.А., а с 0
        self.I0 = []
        self.I1 = []
        self.Id = []
        self.eq = []
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
            else:
                self.eq.append(i)
            i += 1

    def make_matrix(self):
        """
        Создаем матрицу ограничений и вектор свободных
        членов для применения симплекс-метода в методе s_upd
        (создаем задачу линейного программирования)"""
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

        for i in self.eq:  # последние строчки - линейные ограничения - равентсва
            line = self.phi_list[i].grad(self.x)
            line.append(0.0)
            matrix.append(line.copy())
            b.append(0)

        c = [0.0 for i in line0]
        c[-1] = 1.0

        eq_less = 1 + len(self.Id) + 2 * (len(line0) - 1)  # phi0, phi_i, -1 < s_j < 1

        return matrix, b, c, eq_less

    def s_upd(self):
        """
        Находим направление спуска
        """
        M, b, c, eq_less = self.make_matrix()
        x_any = len(c)
        A2, b2, Ind2 = Make_Canon_Form(np.array(M).copy(),
                                       np.array(b).copy(),
                                       False,
                                       0,
                                       eq_less,
                                       0)

        c2, c_free2 = Update_C(A2.copy(), b2.copy(), np.array(c).copy(), 0, Ind2, eq_less, 0, x_any)

        opt_vector, self.eta = Simplex_With_Init(A2.copy(), b2.copy(), Ind2.copy(), c2.copy(), c_free2.copy())
        self.s = Recover_Initial_Variables(opt_vector, x_any - 1)

    def lmd_upd(self):
        K = 0
        R = self.f0.R
        for i in range(len(self.phi_list)):
            if (i not in self.Id) & (self.phi_list[i].K > K):
                K = self.phi_list[i].K
            if (i in self.Id) & (self.phi_list[i].R > R):
                R = self.phi_list[i].R

        lmd_0 = -0.5 * self.eta / (self.f0.R * norma_calculate(self.s) ** 2)
        lmd_d = -1 * self.eta / (R * norma_calculate(self.s) ** 2)
        lmd_nd = self.dlt / (K * norma_calculate(self.s))

        self.lmd = min(lmd_0, lmd_d, lmd_nd)

    def x_upd(self):
        if self.eta < -self.dlt:  # сюда не должен заходить случай с lmd = none
            self.x = [self.x[i] + self.lmd * self.s[i] for i in range(len(self.s))]
        else:
            self.dlt = self.alfa * self.dlt

    def solve_matrix_by_linear_constraints(self):
        """
        метод используется для построения начального приближения find_x0
        """
        matrix = []
        b = []
        x0 = [0.0 for i in self.x]

        for i in self.eq:  # создаем матрицу по линейным ограничениям - равенствам
            line = self.phi_list[i].grad(self.x)
            matrix.append(line.copy())
            b.append(-1 * self.phi_list[i].phi(x0))  # свободный член в линейном уравнении - это phi(0)

        comb = list(combinations(range(len(x0)), len(self.eq)))  # варианты (индексы столбцов) квадратных матриц
        for item in comb:  # пытаемся решить квадратную матрицу. Как только решили => stop
            square_matrix = np.matrix(matrix)[:, list(item)]
            try:
                ans = np.linalg.solve(square_matrix, b)
                for ind in item:
                    x0[ind] = ans[ind]
                break
            except:
                pass
        return x0

    def find_x0(self):
        """
        Находим начальное приближение self.x
        Потом надо будет убрать из конструктора x0
        1) Находим x0
        2) Строим задачу по методу Зойтендейка (создаем новые ограничения p_list на основе предыдущих + целевая ф-я)
        3) Решаем, пока значение целевой функции не станет < 0
        """
        x0 = self.solve_matrix_by_linear_constraints()
        if len(self.phi_list) != 0:  # то есть хоть одно ограничение есть
            eta0 = max([self.phi_list[i].phi(x0) for i in self.Id + self.I1])
            if eta0 < 0:  # если все уже хорошо, то дальше можно не идти
                return x0
        else:  # безусловная минимизация
            return x0

        f = Target_function(lambda x: x[-1],
                            lambda x: [0 if i != len(x) - 1 else 1 for i in range(len(x))])

        p_list = []
        for i in range(len(self.phi_list)):
            type_ = self.phi_list[i].type
            if type_ == 'eq':
                p_list.append(0)
                p_list[i] = Constraint(self.phi_list[i].type,
                                       lambda y, i=i: self.phi_list[i].phi(y[:-1]),
                                       lambda y, i=i: self.phi_list[i].grad(y[:-1]) + [0.0])  # имеется в виду конкатенация
            else:
                p_list.append(0)
                p_list[i] = Constraint(self.phi_list[i].type,
                                       lambda y, i=i: self.phi_list[i].phi(y[:-1]) - y[-1],
                                       lambda y, i=i: self.phi_list[i].grad(y[:-1]) + [-1.0])  # имеется в виду конкатенация

        super_task = Zoitendeik_step(f, p_list, x0 + [eta0], 0.25, 0.5)
        while super_task.f0.f(super_task.x) >= 0:
            super_task.I_upd()
            super_task.s_upd()
            super_task.lmd_upd()
            super_task.x_upd()

        return super_task.x[:-1]

    def minimize(self, eps=0.001):
        self.I_upd()
        self.x = self.find_x0()
        self.print_params()
        i = 0
        while True:  # поставить нормальное условие
            print('N:', i)

            self.I_upd()
            self.s_upd()
            self.lmd_upd()
            self.print_params()
            self.x_upd()

            if self.eta == 0 and self.dlt < eps and self.dlt < -1 * max([self.phi_list[j].phi(self.x) for j in self.Id]):
                print('THE END by ETA = 0')
                break

            if -self.eta < eps and self.dlt < eps:  # если решение - внутри области
                print('THE END by POINT INSIDE SUSPECT')
                break

            i += 1
            if i == 200:  # условие на случай, если оптимум не заключен между двумя ограничениями
                print('THE END by i')
                break

        return self.f0.f(self.x)

    def print_params(self):
        print('x', [round(x_, 5) for x_ in self.x])
        print('f0', round(self.f0.f(self.x), 5))
        print('way', [round(w_, 5) for w_ in self.s])
        print('lmd', self.lmd)
        print('dlt', self.dlt)
        print('eta', self.eta)
        print('I_d', self.Id)
        print()


if __name__ == '__main__':
    phi0 = Target_function(lambda x: 2 * x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 4 * x[0] - 6 * x[1],
                           lambda x: [4 * x[0] - 2 * x[1] - 4, 4 * x[1] - 2 * x[0] - 6])

    phi1 = Constraint('ineq',
                      lambda x: x[0] + 5 * x[1] - 5,
                      lambda x: [1, 5])

    phi2 = Constraint('ineq',
                      lambda x: 2 * x[0] ** 2 - x[1],
                      lambda x: [4 * x[0], -1])

    phi3 = Constraint('ineq',
                      lambda x: -x[0],
                      lambda x: [-1, 0])

    phi4 = Constraint('ineq',
                      lambda x: -x[1],
                      lambda x: [0, -1])

    z = Zoitendeik_step(phi0, [phi1, phi2, phi3, phi4], [0.0, 0.75], 0.25, 0.5)

    z.minimize()

    q0 = Target_function(lambda x: 6 * x[0] ** 2 + x[1] ** 2 - 2 * x[0] * x[1] - 10 * x[1],
                         lambda x: [12 * x[0] - 2 * x[1], 2 * x[1] - 2 * x[0] - 10],
                         R=10)

    q1 = Constraint('eq',
                    lambda x: 2 * x[0] + 0.5 * x[1] - 4,
                    lambda x: [2.0, 0.5],
                    K=2, R=2)

    q2 = Constraint('ineq',
                    lambda x: -2 * x[0] - 2 * x[1] + 5,
                    lambda x: [-2, -2],
                    K=2, R=2)

    q3 = Constraint('ineq',
                    lambda x: -x[0],
                    lambda x: [-1, 0],
                    K=2, R=2)

    q4 = Constraint('ineq',
                    lambda x: -x[1],
                    lambda x: [0, -1],
                    K=2, R=2)

    # здесь решение находится внутри области, поэтому
    # eta никогда не равно нулю, а изменение dlt
    # происходит просто благодаря приближению точки к оптимуму
    qq = Zoitendeik_step(q0, [q1, q2, q3, q4], [2.0, 0.0], 0.5, 0.5)

    # qq.minimize()

    print('cat')
