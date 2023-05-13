import numpy as np
from prettytable import PrettyTable
from simplex_lib.preprocessing import Update_C


def step_print(A, b, c, c0, base_indexes):

    A_print = np.zeros((A.shape[0] + 2) * (A.shape[1] + 2)).reshape(A.shape[0] + 2, A.shape[1] + 2)
    A_print = A_print.astype(str)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_print[i][j] = A[i][j]
        A_print[i, -2] = '='
        A_print[i, -1] = b[i]
    for j in range(A.shape[1]):
        A_print[-2][j] = ' '
        A_print[-1][j] = c[j]
    A_print[-1, -1] = -c0
    A_print[-1, -2] = '->'
    A_print[-2, [-2, -1]] = ' '

    table = PrettyTable()
    table.header = False
    table.add_rows(A_print)
    print('Матрица ограничений:')
    print(table)
    print('Базисные индексы:')
    print(base_indexes)
    print('\n\n')


def create_list_for_leader_line(b_free_chlens, leader_column, B_index_ones, A_matrix):
    """
    Формируем индекс ведущей строки по табличному Симплекс - Методу
    Если найти такую строку не удается, то решение не ограничено
    """
    try:
        result = {j: b_free_chlens[j] / A_matrix[j][leader_column] for j in [k for k in range(A_matrix.shape[0]) if
                                                                             A_matrix[k][leader_column] > 0]}
        filtered_result = {key: val for key, val in result.items() if val >= 0}
        leader_line = min(filtered_result, key=filtered_result.get)
        B_index_ones[leader_line] = leader_column
        return leader_line
    except:
        print("\nСИМПЛЕКС. РЕШЕНИЕ НЕОГРАНИЧЕНО!!!\n")
        return None


def Gauss_Jordano(leader_column, leader_line, leader_value, A_matrix, b_free_chlens, c_deal_func, c_deal_free_chlen):
    """
    Функция каждую строку складывает с ведущей строкой, умноженной на необходимый кожффициент по 
    методу Гаусса
    Обновляется вектор с
    """
    Indexes_values = [j for j in range(0, leader_line)] + [i for i in range(leader_line + 1, A_matrix.shape[0])]
    try:
        for i in Indexes_values:
            coeff = A_matrix[i][leader_column] / leader_value
            for j in range(A_matrix.shape[1]):
                A_matrix[i][j] -= coeff * A_matrix[leader_line][j]
            b_free_chlens[i] -= coeff * b_free_chlens[leader_line]
        coeff = c_deal_func[leader_column] / leader_value
        c_deal_func[:] -= coeff * A_matrix[leader_line][:]
        c_deal_free_chlen -= coeff * b_free_chlens[leader_line]
        for j in range(A_matrix.shape[1]):
            A_matrix[leader_line][j] = A_matrix[leader_line][j] / leader_value
        b_free_chlens[leader_line] = b_free_chlens[leader_line] / leader_value
        return A_matrix, b_free_chlens, c_deal_func, c_deal_free_chlen
    except:
        return None


def beautiful_print_solution(A_matrix, B_index_ones, b_free_chlens, C_deal_free_chlen, C_deal_func):
    """
    вывод решения в консоль
    """
    zero_vector = np.zeros(A_matrix.shape[1])
    for i in range(len(B_index_ones)):
        zero_vector[B_index_ones[i]] = b_free_chlens[i]
    # print('ОПТИМАЛЬНОЕ БАЗИСНОЕ РЕШЕНИЕ ----↓\n', zero_vector)
    # # print('\n МАРИЦА А ----↓ \n', A_matrix)
    # print('\n БАЗИСНЫЕ ИНДЕКСЫ ----↓ \n', B_index_ones)
    # print('\n СТОЛБЕЦ СВОБОДНЫХ ЧЛЕНОВ ----↓ \n', '', b_free_chlens)
    # print('\n ВЕКТОР С ----↓ \n', '', C_deal_func)
    # print('\n ЗНАЧЕНИЕ ЦЕЛЕВОЙ ФУНКЦИИ ----↓ \n', '', -C_deal_free_chlen)
    # print('\n\n')

    return zero_vector, -C_deal_free_chlen


def pivot(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen, leader_column, leader_line):
    """
    Функция pivot - см. Корман
    """
    leader_value = A_matrix[leader_line][leader_column]
    try:
        A_matrix, b_free_chlens, C_deal_func, C_deal_free_chlen = Gauss_Jordano(leader_column, leader_line,
                                                                            leader_value, A_matrix, b_free_chlens,
                                                                            C_deal_func, C_deal_free_chlen)
        return A_matrix, b_free_chlens, C_deal_func, C_deal_free_chlen
    except:
        return None


def Simplex(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen):
    """
    Функция Simplex - см. Корман Simplex без инициализации
    """
    while not all(x >= -10**(-15) for x in C_deal_func):
        leader_column = min(range(len(C_deal_func)), key=C_deal_func.__getitem__)
        leader_line = create_list_for_leader_line(b_free_chlens, leader_column, B_index_ones, A_matrix)
        try:
            A_matrix, b_free_chlens, C_deal_func, C_deal_free_chlen = \
                pivot(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen, leader_column, leader_line)
        except:
            return None
        # step_print(A_matrix, b_free_chlens, C_deal_func, C_deal_free_chlen, B_index_ones)

    return beautiful_print_solution(A_matrix, B_index_ones, b_free_chlens, C_deal_free_chlen, C_deal_func)

    # return A_matrix, b_free_chlens, C_deal_func, C_deal_free_chlen


def Simplex_With_Init(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen):
    """
    Функция Simplex - см. Корман Simplex
    """
    try:
        A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen = \
            Initialize_Simplex(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen)
        # print('\n---СИМПЛЕКС - МЕТОД---\n')
        return Simplex(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen)
    except:
        return None


def Initialize_Simplex(A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen):
    """
    Функция Initialize_Simplex - см. note8 (не по Кормену)
    """
    # print('\n---ИНИЦИАЛИЗАЦИЯ СИМПЛЕКС-МЕТОДА. ПРОВЕРКА СУЩЕСТВОВАНИЯ РЕШЕНИЯ---\n')
    b_negative_index = list()
    for i in range(A_matrix.shape[0]):
        if b_free_chlens[i] < 0:
            b_negative_index += [i]

    if len(b_negative_index) == 0:
        return A_matrix, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen
    else:
        """добавили aux vars"""
        A_aux = A_matrix
        for i in b_negative_index:
            assrt = np.zeros(A_aux.shape[0])
            assrt[i] = -1
            A_aux = np.insert(A_aux, A_aux.shape[1], assrt, axis=1)
            """нормализуем"""
            A_aux[i, :] = -A_aux[i, :]
            b_free_chlens[i] = -b_free_chlens[i]

        """надо поменять базисный вектор"""
        tmp = len(b_negative_index)
        for i in b_negative_index:
            B_index_ones[i] = A_aux.shape[1] - tmp
            tmp -= 1

        """вектор с"""
        c_aux = np.zeros(A_aux.shape[1])
        for i in range(len(b_negative_index)):
            c_aux[-(i + 1)] = 1
        c0_aux = 0

        """складываем первые строчки по алгоритму"""
        for i in b_negative_index:
            coef = c_aux[B_index_ones[i]] / A_aux[i, B_index_ones[i]]
            c_aux[:] = c_aux[:] - coef * A_aux[i, :]
            c0_aux = c0_aux - coef*b_free_chlens[i]

        """
        Теперь сам симплекс-метод
        B_index_ones сам меняется, хотя мы его и не передаем
        """
        A_aux, b_free_chlens, c_aux, c0_aux = Simplex(A_aux, b_free_chlens, B_index_ones, c_aux, c0_aux)

        if c0_aux == 0:
            print("\nЭТАП ИНИЦИАЛИЗАЦИИ ПРОЙДЕН. ЗАДАЧА ИМЕЕТ РЕШЕНИЕ\n")
            for i in range(len(b_negative_index)):
                A_aux = np.delete(A_aux, [-1], axis=1)
            C_deal_func, C_deal_free_chlen = Update_C(A_aux, b_free_chlens, C_deal_func, C_deal_free_chlen, B_index_ones, 0, 0, 0)

            return A_aux, b_free_chlens, B_index_ones, C_deal_func, C_deal_free_chlen
        else:
            print("\nСИМПЛЕКС. ЭТАП ИНИЦИАЛИЗАЦИИ. ЗАДАЧА НЕРАЗРЕШИМА!!!\n")
            return None


def Recover_Initial_Variables(opt_solution_vector, X_any):
    """
    Предполагаем, что переменные любого знака - все в начале;
    в конце только положиетльные переменные.
    Здесь есть противоречие с тем, как мы приняли вводить данные (сначала вводились толькл + переменные)
    Поэтому не использовать этот метод необдуманно! Либо изменить Make_Canon_Form!
    """
    initial_variables = []
    for i in range(0, 2 * X_any, 2):
        initial_variables.append(opt_solution_vector[i] - opt_solution_vector[i+1])
        i += 2
    return initial_variables




if __name__ == '__main__':
    def main():
        pass
