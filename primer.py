
# Целевая функция: минимизировать общее число врачей
c = [1, 1, 1, 1, 1, 1]

# Матрица ограничений (6x6)
A = [
    [1, 0, 0, 0, 0, 1],  # x6 + x1 >= 40
    [1, 1, 0, 0, 0, 0],  # x1 + x2 >= 30
    [0, 1, 1, 0, 0, 0],  # x2 + x3 >= 20
    [0, 0, 1, 1, 0, 0],  # x3 + x4 >= 50
    [0, 0, 0, 1, 1, 0],  # x4 + x5 >= 80
    [0, 0, 0, 0, 1, 1]   # x5 + x6 >= 100
]

b = [40, 30, 20, 50, 80, 100]

# Знаки ограничений
signs = [">=", ">=", ">=", ">=", ">=", ">="]

# 1. Парсим
problem = parse_problem(c, A, b, signs)

# 2. Каноническая форма (slack/surplus, без искусственных)
A_can, b_can, var_names, basis = to_canonical(problem)

# 3. Добавляем искусственные переменные (v1, v2, ...)
A_with_art, b_with_art, var_names_art, basis_art, art_vars = add_artificial_vars(
    A_can, b_can, signs, var_names, basis
)

print_tableau_df(A_with_art, b_with_art, var_names_art, basis_art)
print("Artificial variable indices:", art_vars)
# 4. Печатаем таблицу Phase I
print_phase1_tableau(A_with_art, b_with_art, var_names_art, basis_art, art_vars)



################################2

c = [2, 3]  # минимизируем 

# матрица ограничений (каждая строка — коэффициенты при x1,x2)
A = [
    [1,  2],   # x1 + 2x2 <= 4
    [2,  1],   # 2x1 + x2 >= 3
    [1, -1]    # x1 - x2 = 1
]

b = [4, 3, 1]

signs = ["<=", ">=", "="]


# 1. Парсим
problem = parse_problem(c, A, b, signs)

# 2. Каноническая форма (slack/surplus, без искусственных)
A_can, b_can, var_names, basis = to_canonical(problem)

# 3. Добавляем искусственные переменные (v1, v2, ...)
A_with_art, b_with_art, var_names_art, basis_art, art_vars = add_artificial_vars(
    A_can, b_can, signs, var_names, basis
)

print_tableau_df(A_with_art, b_with_art, var_names_art, basis_art)
print("Artificial variable indices:", art_vars)
# 4. Печатаем таблицу Phase I
print_phase1_tableau(A_with_art, b_with_art, var_names_art, basis_art, art_vars)


#############################3
# ==== Пример ====
data = [
    [1,  2,  1,  0,  0,  0,  4],   # s1
    [2,  1,  0, -1,  1,  0,  3],   # v1
    [1, -1,  0,  0,  0,  1,  1],   # v2
    [-3, 0,  0,  1, -1, -1, -4]    # w
]

columns = ["x1", "x2", "s1", "s2", "v1", "v2", "RHS"]
index = ["s1", "v1", "v2", "w"]

df = pd.DataFrame(data, columns=columns, index=index, dtype=float)
basis = ["s1", "v1", "v2"]

df_final, basis_final = phase1_simplex(df, basis)