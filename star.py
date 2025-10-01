
def to_canonical(problem):
    """Каноническая форма: добавляем slack/surplus переменные, но без искусственных"""
    A, b, signs = problem["A"], problem["b"], problem["signs"]
    m, n = len(A), len(A[0])
    A_new = []
    var_names = [f"x{j+1}" for j in range(n)]
    basis = []
    
    slack_count = 0
    for i in range(m):
        row = A[i][:]
        if signs[i] == "<=":
            slack = [0]*slack_count + [1]
            slack += [0]*(m - len(slack) - 1)
            row += slack
            var_names.append(f"s{slack_count+1}")
            basis.append(var_names[-1])
            slack_count += 1
        elif signs[i] == ">=":
            # surplus -s
            slack = [0]*slack_count + [-1]
            slack += [0]*(m - len(slack) - 1)
            row += slack
            var_names.append(f"s{slack_count+1}")
            slack_count += 1
        elif signs[i] == "=":
            # равенство – без слэка
            row += [0]*slack_count
        A_new.append(row)
    
    return A_new, b, var_names, basis


def add_artificial_vars(A, b, signs, var_names, basis):
    """Добавляем искусственные переменные для Phase I"""
    m = len(A)
    n = len(A[0])
    A_new = [row[:] for row in A]
    var_names_new = var_names[:]
    artificial_vars = []

    for i in range(m):
        if signs[i] in (">=", "="):
            col = [0]*m
            col[i] = 1
            for r in range(m):
                A_new[r].append(col[r])
            var_names_new.append(f"v{len(artificial_vars)+1}")
            artificial_vars.append(len(var_names_new)-1)
            basis.append(var_names_new[-1]) 

    return A_new, b, var_names_new, basis, artificial_vars



def print_tableau_df(A, b, var_names, basis):
    data = []
    n_vars = len(var_names)  # <- используем именно эту длину
    for i, row in enumerate(A):
        # Если длина строки меньше, чем количество имен, дополняем нулями
        row_extended = row + [0]*(n_vars - len(row))
        row_data = {var_names[j]: row_extended[j] for j in range(n_vars)}
        row_data["RHS"] = b[i]
        row_data["Basis"] = basis[i] if i < len(basis) else "-"
        data.append(row_data)
    df = pd.DataFrame(data)
    df = df[["Basis"] + var_names + ["RHS"]]
    print(df)


    