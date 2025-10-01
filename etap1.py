import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def parse_problem(c, A, b, signs):
    """
    Упаковывает задачу в словарь
    c: список коэффициентов целевой функции (минимизация)
    A: матрица ограничений (списки)
    b: свободные члены
    signs: список знаков ('<=', '>=', '=')
    """
    return {
        "c": c[:],
        "A": [row[:] for row in A],
        "b": b[:],
        "signs": signs[:]
    }


def to_canonical(problem):
    """
    Преобразуем A-> каноническую (добавляем слэк/сурплус),
    возвращает:
    A_new — новая матрица с slack/surplus,
    b — правые части,
    var_names — список всех переменных,
    basis_per_row — имя базисной переменной для каждой строки (или None).
    """
    A, b, signs = problem["A"], problem["b"], problem["signs"]
    m = len(A)
    n = len(A[0])

    # сколько слэков/сюрплусов всего нужно
    slack_needed = [1 if s in ("<=", ">=") else 0 for s in signs]
    total_slacks = sum(slack_needed)

    # имена переменных: x1..xn, затем s1..s_total
    var_names = [f"x{j+1}" for j in range(n)] + [f"s{j+1}" for j in range(total_slacks)]

    A_new = []
    basis_per_row = [None] * m
    slack_idx = 0

    for i in range(m):
        row = list(A[i])
        slacks = [0.0] * total_slacks
        if signs[i] == "<=":
            # slack +1, входит в базис
            slacks[slack_idx] = 1.0
            basis_per_row[i] = f"s{slack_idx+1}"
            slack_idx += 1
        elif signs[i] == ">=":
            # surplus -1 (не базисная)
            slacks[slack_idx] = -1.0
            slack_idx += 1
        else:  # "="
            # нет доб. перем.
            pass
        row += slacks
        A_new.append(row)

    return A_new, b, var_names, basis_per_row


def add_artificial_vars(A, b, signs, var_names, basis_per_row):
    """
    Добавляем искусственные переменные v_i. 
    basis_per_row — список имён базисных переменных для строк (может содержать None).
    Возвращает basis_per_row заполненным (имена), и artificial_vars как список имён.
    """
    m = len(A)
    A_new = [row[:] for row in A]
    var_names_new = var_names[:]
    artificial_vars = []

    for i in range(m):
        if signs[i] in (">=", "="):
            # добавить столбец искусственной, 1 в строке i
            for r in range(m):
                A_new[r].append(1.0 if r == i else 0.0)
            v_name = f"v{len(artificial_vars)+1}"
            var_names_new.append(v_name)
            artificial_vars.append(v_name)
            # искусственная — базисная переменная для этой строки
            basis_per_row[i] = v_name

    return A_new, b, var_names_new, basis_per_row, artificial_vars


def align_matrix(A, var_names):
    """Расширяет матрицу A нулями до количества переменных"""
    cols_needed = len(var_names)
    for i in range(len(A)):
        if len(A[i]) < cols_needed:
            A[i] = A[i] + [0] * (cols_needed - len(A[i]))
    return A


def print_df_rich(df):
    """Красивый вывод DataFrame через rich, с сохранением индекса"""
    pd.set_option("display.max_columns", None)    
    pd.set_option("display.width", 2000)          
    pd.set_option("display.precision", 3)         
    pd.set_option("display.float_format", "{:>8.1f}".format)  
    pd.set_option("display.max_colwidth", None)
    table = Table(show_header=True, header_style="bold cyan")

    # первая колонка — индекс
    table.add_column("Basis", justify="left")

    # остальные колонки
    for col in df.columns:
        table.add_column(str(col), justify="right")

    # строки
    for i, (idx, row) in enumerate(df.iterrows()):
        # подсветка Basis светло-зелёным
        basis_val = f"[bright_green]{idx}[/bright_green]"
        row_vals = [f"{val:.1f}" if isinstance(val, (int, float)) else str(val) for val in row.values]
        # если последняя строка
        if i == len(df) - 1:
            row_vals = [f"[pale_turquoise4]{v}[/pale_turquoise4]" for v in row_vals]

        table.add_row(basis_val, *row_vals)
    console.print(table)


def print_tableau_df(A, b, var_names, basis):
    data = []
    n_vars = len(var_names)
    for i, row in enumerate(A):
        row_extended = row + [0]*(n_vars - len(row))
        row_data = {var_names[j]: row_extended[j] for j in range(n_vars)}
        row_data["RHS"] = b[i]
        row_data["Basis"] = basis[i] if i < len(basis) else "-"
        data.append(row_data)

    # таблица rich
    table = Table(title="Симплекс-таблица", show_lines=True)

    # заголовки
    table.add_column("Basis", style="cyan", justify="center")
    for var in var_names:
        table.add_column(var, style="magenta", justify="center")
    table.add_column("RHS", style="green", justify="center")

    # строки
    for row in data:
        table.add_row(
            row["Basis"],
            *[str(row[var]) for var in var_names],
            str(row["RHS"])
        )

    console.print(table)


def align_matrix(A, var_names):
    """Расширяет матрицу A нулями до количества переменных"""
    cols_needed = len(var_names)
    for i in range(len(A)):
        if len(A[i]) < cols_needed:
            A[i] = A[i] + [0] * (cols_needed - len(A[i]))
    return A


def phase1_objective(A, b, var_names, basis, artificial_vars):
    """
    Считает строку W как -(sum строк, где базисная переменная — искусственная).
    Возвращает (obj, rhs) — список коэффициентов obj по var_names и rhs.
    """
    m = len(A)
    n = len(var_names)
    obj = [0.0] * n
    rhs = 0.0

    for i, bi in enumerate(basis):
        if bi in artificial_vars:
            # вычитаем i-ю строку из obj (w = -(sum rows_with_art))
            row = A[i]
            for j in range(n):
                obj[j] -= row[j]
            rhs -= float(b[i])

    return obj, rhs




def print_phase1_tableau(A, b, var_names, basis_per_row, artificial_vars):
    """
    basis_per_row: список длины m, элемент i — базисная переменная для строки i
    """
    A = align_matrix(A, var_names)
    obj, rhs = phase1_objective(A, b, var_names, basis_per_row, artificial_vars)

    n_vars = len(var_names)
    data = []

    # строки ограничений
    for i, row in enumerate(A):
        row_extended = row + [0]*(n_vars - len(row))
        row_data = {var_names[j]: float(row_extended[j]) for j in range(n_vars)}
        row_data["RHS"] = float(b[i])
        row_data["Basis"] = basis_per_row[i] if i < len(basis_per_row) else "-"
        data.append(row_data)

    # строка цели Phase I
    obj_extended = obj + [0]*(n_vars - len(obj))
    row_data = {var_names[j]: float(obj_extended[j]) for j in range(n_vars)}
    row_data["RHS"] = float(rhs)
    row_data["Basis"] = "w"
    data.append(row_data)

    df = pd.DataFrame(data)
    df = df.set_index("Basis")
    df = df[var_names + ["RHS"]].astype(float)  # только числа для вычислений

    # --- проверка соответствия базиса ---
    eps = 1e-9
    for i, name in enumerate(basis_per_row):
        if name not in df.columns:
            raise RuntimeError(f"Базисная переменная {name} отсутствует в столбцах.")
        col = df[name].to_numpy()[:-1]  # исключаем w
        if not (abs(col[i] - 1.0) < eps and all(abs(col[j]) < eps for j in range(len(col)) if j != i)):
            raise RuntimeError(f"Несоответствие базиса для {name} в строке {i}.\nСтолбец:\n{col}")
    # -------------------------------------

    print_df_rich(df)
    return df


##########################

def simplex_iteration_df(df, basis):
    """
    Делает одну итерацию симплекс-метода (Phase I) с DataFrame.
    df: DataFrame, где строки = ограничения + последняя строка w.
    basis: список базисных переменных.
    """

    m, n = df.shape
    var_names = list(df.columns[:-1])  # все переменные, кроме RHS

    # 1. Находим входящую переменную (самый отрицательный в строке w)
    w_row = df.loc["w", var_names]
    entering = w_row.idxmin()

    if w_row[entering] >= 0:
        # Решение найдено
        return df, basis, None, None, True

    # 2. Находим выходящую переменную
    ratios = []
    for row in df.index[:-1]:  # кроме w
        a_ij = df.loc[row, entering]
        if a_ij > 0:
            ratios.append((df.loc[row, "RHS"] / a_ij, row))
    if not ratios:
        raise ValueError("Задача неограничена")

    leaving = min(ratios, key=lambda x: x[0])[1]

    # 3. Пивот
    pivot = df.loc[leaving, entering]
    df.loc[leaving, :] /= pivot

    for row in df.index:
        if row != leaving:
            factor = df.loc[row, entering]
            df.loc[row, :] -= factor * df.loc[leaving, :]

    # 4. Обновляем базис
    basis[basis.index(leaving)] = entering
    df = df.rename(index={leaving: entering})

    return df, basis, entering, leaving, False


def phase1_simplex(df, basis):
    """
    Запускает Phase I до конца.
    df: DataFrame с начальным симплекс-табло.
    basis: список базисных переменных.
    """

    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Итерация {iteration} ===")
        print_df_rich(df)
        #print(df)

        df, basis, entering, leaving, finished = simplex_iteration_df(df, basis)

        if finished:
            print("\nФаза I завершена ✅")
            break
        else:
            print(f"\nВходит: {entering}, выходит: {leaving}")
            print("Новый базис:", basis)

    return df, basis

#####################################################3


def run_phase1(c, A, b, signs):
    # 1. Парсим
    problem = parse_problem(c, A, b, signs)

    # 2. Каноническая форма (slack/surplus, без искусственных)
    A_can, b_can, var_names, basis = to_canonical(problem)

    # 3. Добавляем искусственные переменные vi
    A_can, b_can, var_names, basis_per_row = to_canonical(problem)
    A_art, b_art, var_names_art, basis_per_row, artificial_vars = add_artificial_vars(
    A_can, b_can, problem["signs"], var_names, basis_per_row
)
    
    # 4. Строим начальный DataFrame Phase I
    df = print_phase1_tableau(A_art, b_art, var_names_art, basis_per_row, artificial_vars)

    print("\n--- DataFrame для Phase I (перед итерациями) ---")
    print_df_rich(df)
    #print(df)
    print("Базис:", basis_per_row)

    # 5. Цикл итераций Phase I
    df_final, basis_final = phase1_simplex(df, basis_per_row)
    return df_final, basis_final

################################################
################################################
def prepare_phase2(df, basis, c):
    """
    Удаляет искусственные переменные и формирует таблицу для Фазы II (минимизация).
    df    — таблица после Фазы I
    basis — список базисных переменных
    c     — исходные коэффициенты целевой функции (для минимизации)
    """
    # 1. Убираем искусственные переменные (столбцы, начинающиеся с 'v')
    cols_to_drop = [col for col in df.columns if str(col).startswith("v")]
    df = df.drop(columns=cols_to_drop)

    # 2. Заменяем строку цели (W → Z)
    n_vars = len(c)
    new_obj = {col: 0 for col in df.columns}
    for j, coeff in enumerate(c):
        var = f"x{j+1}"
        if var in new_obj:
            new_obj[var] = coeff

    # Добавим RHS для цели
    new_obj["RHS"] = 0
    df.loc["Z"] = new_obj

    # 3. Пересчёт строки цели
    # c'_j = c_j - sum(c_b * a_ij),   Z = Z - sum(c_b * b_i)
    for var in basis:
        if var.startswith("x"):  
            c_b = c[int(var[1:]) - 1]         # коэффициент при базисной переменной
            df.loc["Z"] -= c_b * df.loc[var]  # Вычитаем c_b * строка базисной переменной
    
    print("Таблица после подготовки к Фазе II:")
    print_df_rich(df)
    #print(df)
    return df, basis


def run_phase2(df, basis):
    """
    Фаза II: обычный симплекс для минимизации.
    """
    print("\n=== Фаза II: запуск симплекса ===")
    result = simplex_iteration_df(df, basis)

    # если simplex_iteration_df возвращает кортеж
    if isinstance(result, tuple):
        final_df, final_basis, *rest = result
    else:
        final_df, final_basis = result, basis

    print("\nФаза II завершена ✅")
    print("Финальная таблица:")
    print_df_rich(final_df)
    #print(final_df)

    return final_df, final_basis



def run_to_final(c, A, b, signs):
    """
    Полный цикл: Фаза I -> Фаза II -> финальное решение.
    """
    df, basis = run_phase1(c, A, b, signs)

    if df.loc["w", "RHS"] > 1e-9:
        print("Задача несовместна (нет допустимого решения).")
        return None, None
 
    df2, basis2 = prepare_phase2(df, basis, c)
    final_df, final_basis = run_phase2(df2, basis2)

    print("\n=== Оптимальное решение ===")
    # значения переменных
    solution = {col: 0 for col in final_df.columns if col not in ["RHS"]}
    for var in final_basis:
        solution[var] = float(final_df.loc[var, "RHS"])

    print("Переменные:", solution)

    # значение целевой функции
    z_val = -final_df.loc["Z", "RHS"]
    print("Оптимальное значение Z:", z_val)

    return final_df, final_basis