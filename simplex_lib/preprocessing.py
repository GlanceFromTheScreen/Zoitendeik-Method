import numpy as np


def Find_Not_Zero_And_Swap(M, first_row_to_start, k, n, Basis_Indexes):
    """
    if we have the case, when we need to divide on 0 in Gauss method, we need to change columns to find not 0 coef.
    """
    flag = False
    if M[k + first_row_to_start, Basis_Indexes[k]] == 0:
        position_in_basis_indexes = 0
        for w in Basis_Indexes:
            if M[k + first_row_to_start, w] != 0:
                Basis_Indexes[k], Basis_Indexes[position_in_basis_indexes] = Basis_Indexes[position_in_basis_indexes], Basis_Indexes[k]
                flag = True
                return True
            position_in_basis_indexes += 1
        if not flag:
            return False
    return True


def Gauss_One_Step(M, b_vector, n, i, t, p):
    """
    прибавляет к i-й строке t-ю строку с нужным коэффициентом из столбца p (метод Гаусса)
    """
    coef = M[i, p] / M[t, p]
    for j in range(0, n, 1):
        M[i, j] = M[i, j] + (-coef) * M[t, j]
        M[i, j] = '{:.8f}'.format(M[i, j])
    b_vector[i] = b_vector[i] + (-coef) * b_vector[t]


def Mini_Gauss(M, n, m, b_vector, first_row_to_start, Basis_Indexes):
    """
    Вспомогательная функция.
    Идет форматирование на 1 знак после запятой.
    Функция приводит подматрицу к единичной матрице по методу Гаусса
    :param Basis_Indexes:
    :param M: Матрица на обработку
    :param n: число столбцов M
    :param m: число строк M
    :param b_vector: вектор-столбец свободных членов (размерность m)
    :param first_row_to_start: номер строки, начиная с которого начинается приведение к диаг. виду
    :return: Матрица, у которой в левом нижнем углу стоит единичная матрица
    """

    fake_lines_indexes = []  # строки, которые соответствуют вырожденным базисным переменным

    """forward"""
    for k in range(0, m - first_row_to_start, 1):  # 0, 1
        flag = Find_Not_Zero_And_Swap(M, first_row_to_start, k, n, Basis_Indexes)
        if flag:
            for i in range(first_row_to_start + k + 1, m, 1):  # 4, 5 -- 5
                Gauss_One_Step(M, b_vector, n, i, k + first_row_to_start, Basis_Indexes[k])
        elif (not flag) & (b_vector[k] == 0):
            # print("базис вырожден")
            fake_lines_indexes.append(k + first_row_to_start)  # не уверен, что если f_r_t_s != 0, то все заработает
        else:
            return None

    """back"""
    for k in range(m - first_row_to_start - 1, 0, -1):  # 2, 1
        flag = Find_Not_Zero_And_Swap(M, first_row_to_start, k, n, Basis_Indexes)
        if flag:
            for i in range(first_row_to_start + k - 1, first_row_to_start - 1, -1):  # 4, 3  --  3
                Gauss_One_Step(M, b_vector, n, i, k + first_row_to_start, Basis_Indexes[k])
        elif k in fake_lines_indexes:
            # print("базис вырожден проверка")
            pass
        else:
            return None

    """other coefficients"""
    for k in range(0, m - first_row_to_start, 1):  # 0, 1, 2
        for i in range(0, first_row_to_start, 1):  # 0, 1, 2
            if k + first_row_to_start not in fake_lines_indexes:  # хотя от того, что мы нуди один рар прибавим, ничего не изменится
                Gauss_One_Step(M, b_vector, n, i, k + first_row_to_start, Basis_Indexes[k])

    """Normalize"""
    for i in range(first_row_to_start, m, 1):
        if i not in fake_lines_indexes:
            coef = M[i, Basis_Indexes[i - first_row_to_start]]
            for j in range(0, n, 1):
                M[i, j] = M[i, j] / coef
            b_vector[i] = b_vector[i] / coef

    return M, b_vector


def Make_Canon_Form(Input_Matrix, b, Is_Max, X_Positive, Equation_Less, Equation_More):
    """
    1) Making canon form from input data
    2) Selecting basis: columns, that were added due to inequalities are firstly obtained in basis
       Then we are looking for other basis columns from left to right, trying to make them view like (1, 0, ... 0)
    """

    N_General = Input_Matrix.shape[1]
    M_General = Input_Matrix.shape[0]
    X_Any = N_General - X_Positive

    Total_Restrictions = N_General + X_Any + Equation_Less + Equation_More
    A = np.zeros(M_General * Total_Restrictions).reshape((M_General, Total_Restrictions))
    Basis_Indexes = np.arange(Total_Restrictions)

    """
    заполнение матрицы A переменными из Input, которые
    переводят систему в каноническую форму;
    сначала заполняются переменные без ограничений на знак -
    потом с ограничением
    """
    for i in range(0, X_Positive, 1):
        for j in range(0, M_General, 1):
            A[j, i] = Input_Matrix[j, i]
    for i in range(X_Positive, N_General, 1):
        for j in range(0, M_General, 1):
            A[j, 2 * i - X_Positive] = Input_Matrix[j, i]
            A[j, 2 * i - (X_Positive - 1)] = -Input_Matrix[j, i]

    """
    неравенства типа '>=' переводятся в неравенства '<=' в случае задачи max;
    иначе - наоборот
    """
    if Is_Max:
        for i in range(Equation_Less, Equation_Less + Equation_More, 1):
            A[i, :] = -A[i, :]
            b[i] = -b[i]
    else:
        for i in range(0, Equation_Less, 1):
            A[i, :] = -A[i, :]
            b[i] = -b[i]

    """
    добавление переменных в неравенства, чтобы сделать их равенствами
    """
    if Is_Max:
        for i in range(0, Equation_Less + Equation_More, 1):
            A[i, N_General + X_Any + i] = 1
    else:
        for i in range(0, Equation_Less + Equation_More, 1):
            A[i, N_General + X_Any + i] = -1

    """
    выделение первого, пусть даже и недопустимого базиса - 
    то есть выведение коэффициентов, равных 1, при базисных переменных
    """
    out = Mini_Gauss(A, Total_Restrictions, M_General, b, Equation_Less + Equation_More, Basis_Indexes)
    if out is not None:
        Ind_Final = list(Basis_Indexes[Total_Restrictions - (Equation_More + Equation_Less):Total_Restrictions]) +\
               list(Basis_Indexes[0:M_General - (Equation_More + Equation_Less)])

        """Normalize"""
        for i in range(0, M_General, 1):
            coef = A[i, Ind_Final[i]]
            if coef != 0:
                A[i, :] = A[i, :] / coef
                b[i] = b[i] / coef

        return A, b, Ind_Final

    else:
        return None


def Update_C(M, b, c, c_free, Basis_Indexes, Equation_Less, Equation_More, X_Any):
    """
    Размер матрицы после приведения в каноническую форму мог быть изменен.
    Приведем в соответствие вектор с
    """
    c_reshaped = list(c) + list(np.zeros(Equation_Less + Equation_More + X_Any))
    X_Positive = len(c) - X_Any

    """представление вектора c через новые переменные"""
    for i in range(0, X_Positive, 1):
        c_reshaped[i] = c[i]
    for i in range(X_Positive, X_Positive + X_Any, 1):
        c_reshaped[2 * i - X_Positive] = c[i]
        c_reshaped[2 * i - (X_Positive - 1)] = -c[i]

    """представление вектора с через небазисные компоненты (обнуление базисных компонент)"""
    for i in range(M.shape[0]):
        """в знаменателе обязана быть 1"""
        if M[i, Basis_Indexes[i]] != 0:
            coef = c_reshaped[Basis_Indexes[i]] / M[i, Basis_Indexes[i]]
            c_reshaped[:] = c_reshaped[:] - coef * M[i][:]
            c_free = c_free - coef * b[i]
    return c_reshaped, c_free
