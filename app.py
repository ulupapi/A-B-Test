import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog


# Исходные данные задачи
ROW_LABELS = ["A1 (Коллаборативная)", "A2 (Контентная)"]
COL_LABELS = ["B1 (Активные)", "B2 (Случайные)", "B3 (Пассивные)"]
PAYOFF_MATRIX = np.array(
    [
        [9.0, 17.0, 7.0],
        [15.0, 7.0, 13.0],
    ]
)


def find_dominated_strategies(matrix: np.ndarray):
    """Ищет доминируемые стратегии для игрока A (макс) и игрока B (мин)."""
    m, n = matrix.shape
    dominated_rows = []
    dominated_cols = []

    # Для игрока A: строка i доминирует k, если i >= k покомпонентно и строго > хотя бы в одном месте
    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            if np.all(matrix[i, :] >= matrix[k, :]) and np.any(matrix[i, :] > matrix[k, :]):
                dominated_rows.append((k, i))

    # Для игрока B: столбец j доминирует l, если j <= l покомпонентно и строго < хотя бы в одном месте
    for j in range(n):
        for l in range(n):
            if j == l:
                continue
            if np.all(matrix[:, j] <= matrix[:, l]) and np.any(matrix[:, j] < matrix[:, l]):
                dominated_cols.append((l, j))

    return dominated_rows, dominated_cols


def reduce_by_dominance(matrix: np.ndarray, row_labels: list[str], col_labels: list[str]):
    """Пошагово убирает доминируемые стратегии (если есть)."""
    cur_matrix = matrix.copy()
    cur_rows = row_labels.copy()
    cur_cols = col_labels.copy()
    steps = []

    changed = True
    while changed:
        changed = False
        dom_rows, dom_cols = find_dominated_strategies(cur_matrix)

        if dom_rows:
            # Удаляем первую найденную доминируемую строку
            row_to_remove, row_dominator = dom_rows[0]
            steps.append(
                f"Удалена {cur_rows[row_to_remove]}: доминируется {cur_rows[row_dominator]}"
            )
            cur_matrix = np.delete(cur_matrix, row_to_remove, axis=0)
            del cur_rows[row_to_remove]
            changed = True
            continue

        if dom_cols:
            # Удаляем первый найденный доминируемый столбец
            col_to_remove, col_dominator = dom_cols[0]
            steps.append(
                f"Удалена {cur_cols[col_to_remove]}: доминируется {cur_cols[col_dominator]}"
            )
            cur_matrix = np.delete(cur_matrix, col_to_remove, axis=1)
            del cur_cols[col_to_remove]
            changed = True

    return cur_matrix, cur_rows, cur_cols, steps


def saddle_point_analysis(matrix: np.ndarray):
    """Считает maxmin и minmax."""
    row_mins = matrix.min(axis=1)
    col_maxs = matrix.max(axis=0)
    lower_value = row_mins.max()  # нижняя цена игры
    upper_value = col_maxs.min()  # верхняя цена игры

    return {
        "row_mins": row_mins,
        "col_maxs": col_maxs,
        "lower_value": lower_value,
        "upper_value": upper_value,
        "has_saddle": np.isclose(lower_value, upper_value),
    }


def graphical_solution_2xN(
    matrix_original: np.ndarray,
    matrix_reduced: np.ndarray,
    reduced_col_labels: list[str],
    original_col_labels: list[str],
):
    """
    Графоаналитический метод:
    - строим функции выигрыша по p1 для исходной 2xN матрицы,
    - оптимум считаем по reduced-матрице (обычно после удаления доминируемых стратегий).
    """
    p_grid = np.linspace(0.0, 1.0, 501)
    lines = {}
    for j, col_name in enumerate(original_col_labels):
        values = p_grid * matrix_original[0, j] + (1.0 - p_grid) * matrix_original[1, j]
        lines[col_name] = values

    envelope = np.min(np.vstack(list(lines.values())), axis=0)

    # Точное решение для 2x2 после доминирования
    m_red, n_red = matrix_reduced.shape
    if m_red != 2:
        raise ValueError("Графоаналитический метод здесь реализован для 2xN (две стратегии игрока A).")

    if n_red == 2:
        a11, a12 = matrix_reduced[0, 0], matrix_reduced[0, 1]
        a21, a22 = matrix_reduced[1, 0], matrix_reduced[1, 1]

        den = (a11 - a21) - (a12 - a22)
        if np.isclose(den, 0.0):
            # Параллельные прямые: выбираем лучшее граничное p
            candidates = [0.0, 1.0]
            vals = []
            for p in candidates:
                vals.append(min(p * a11 + (1 - p) * a21, p * a12 + (1 - p) * a22))
            best_idx = int(np.argmax(vals))
            p_opt = candidates[best_idx]
            v_opt = vals[best_idx]
        else:
            p_opt = (a22 - a21) / den
            p_opt = float(np.clip(p_opt, 0.0, 1.0))
            f1 = p_opt * a11 + (1.0 - p_opt) * a21
            f2 = p_opt * a12 + (1.0 - p_opt) * a22
            v_opt = float(min(f1, f2))

        # Для игрока B: q на первой колонке reduced-матрицы
        den_q = (a11 - a12) - (a21 - a22)
        if np.isclose(den_q, 0.0):
            q_first = 0.5
        else:
            q_first = (a22 - a12) / den_q
            q_first = float(np.clip(q_first, 0.0, 1.0))

        q_reduced = np.array([q_first, 1.0 - q_first])
    else:
        # Общий случай 2xN: численный поиск по сетке
        idx = int(np.argmax(envelope))
        p_opt = float(p_grid[idx])
        v_opt = float(envelope[idx])
        q_reduced = np.full(n_red, np.nan)

    # Преобразуем q из reduced в q исходной матрицы
    q_full = np.zeros(len(original_col_labels))
    for idx_red, col_name in enumerate(reduced_col_labels):
        original_idx = original_col_labels.index(col_name)
        q_full[original_idx] = q_reduced[idx_red] if idx_red < len(q_reduced) else 0.0

    return {
        "p_grid": p_grid,
        "lines": lines,
        "envelope": envelope,
        "p_opt": p_opt,
        "p2_opt": 1.0 - p_opt,
        "q_opt": q_full,
        "value": v_opt,
    }


def simplex_solution_for_A(matrix: np.ndarray):
    """ЛП для игрока A: max v, A^T p >= v, sum(p)=1, p>=0."""
    m, n = matrix.shape

    # Вектор переменных: [p1, ..., pm, v]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # max v -> min -v

    # -A^T p + v <= 0
    A_ub = np.hstack([-matrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # sum(p)=1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog (игрок A) не сошелся: {res.message}")

    p = res.x[:m]
    v = res.x[-1]
    return p, float(v), res


def simplex_solution_for_B(matrix: np.ndarray):
    """Двойственная ЛП для игрока B: min v, A q <= v, sum(q)=1, q>=0."""
    m, n = matrix.shape

    # Вектор переменных: [q1, ..., qn, v]
    c = np.zeros(n + 1)
    c[-1] = 1.0  # min v

    # A q - v <= 0
    A_ub = np.hstack([matrix, -np.ones((m, 1))])
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog (игрок B) не сошелся: {res.message}")

    q = res.x[:n]
    v = res.x[-1]
    return q, float(v), res


def brown_robinson(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], iterations: int = 200):
    """
    Метод Брауна–Робинсона.
    Возвращает историю оценок и приближенные смешанные стратегии.
    """
    m, n = matrix.shape

    row_counts = np.zeros(m, dtype=int)
    col_counts = np.zeros(n, dtype=int)

    history = []

    for t in range(1, iterations + 1):
        # Ход A: лучший ответ на частоты B предыдущих шагов
        if t == 1:
            i = int(np.argmax(np.min(matrix, axis=1)))
        else:
            q_prev = col_counts / (t - 1)
            expected_rows = matrix @ q_prev
            i = int(np.argmax(expected_rows))

        row_counts[i] += 1

        # Ход B: лучший ответ на текущие частоты A
        p_curr = row_counts / t
        expected_cols = p_curr @ matrix
        j = int(np.argmin(expected_cols))
        col_counts[j] += 1

        # Верхняя/нижняя оценки цены игры на шаге t
        sum_if_A_pure = matrix @ col_counts
        upper_t = float(np.max(sum_if_A_pure) / t)

        sum_if_B_pure = row_counts @ matrix
        lower_t = float(np.min(sum_if_B_pure) / t)

        value_mid = 0.5 * (upper_t + lower_t)
        gap = upper_t - lower_t

        p_est = row_counts / t
        q_est = col_counts / t

        row = {
            "k": t,
            "A_k": row_labels[i],
            "B_k": col_labels[j],
            "p1": p_est[0],
            "p2": p_est[1] if m > 1 else 0.0,
            "lower_v": lower_t,
            "upper_v": upper_t,
            "v_avg": value_mid,
            "gap": gap,
        }

        for idx_col, name_col in enumerate(col_labels):
            row[f"q_{name_col.split(' ')[0]}"] = q_est[idx_col]

        history.append(row)

    history_df = pd.DataFrame(history)
    p_final = row_counts / iterations
    q_final = col_counts / iterations

    return history_df, p_final, q_final


def payoff_dataframe(matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def main():
    st.set_page_config(page_title="Теория игр: A/B рекомендаций", layout="wide")

    st.title("Стратегия A/B тестирования алгоритма рекомендаций")
    st.caption("Нулевая сумма: Игрок A максимизирует прирост конверсии, Игрок B минимизирует его.")

    st.subheader("1) Исходная матрица и постановка")
    st.markdown(
        """
- Игрок **A (Команда)**: выбирает `A1` (коллаборативная) или `A2` (контентная).
- Игрок **B (Природа/Аналитика)**: выбирает `B1` (активные), `B2` (случайные), `B3` (пассивные).
- Выигрыш в матрице — прирост конверсии Игрока A.
        """
    )
    st.dataframe(payoff_dataframe(PAYOFF_MATRIX, ROW_LABELS, COL_LABELS), use_container_width=True)

    st.subheader("2) Анализ доминирования")
    dominated_rows, dominated_cols = find_dominated_strategies(PAYOFF_MATRIX)

    if dominated_rows:
        for row_to_remove, row_dom in dominated_rows:
            st.write(
                f"Строка **{ROW_LABELS[row_to_remove]}** доминируется строкой **{ROW_LABELS[row_dom]}**."
            )
    else:
        st.write("Для Игрока A доминируемых стратегий нет.")

    if dominated_cols:
        for col_to_remove, col_dom in dominated_cols:
            st.write(
                f"Столбец **{COL_LABELS[col_to_remove]}** доминируется столбцом **{COL_LABELS[col_dom]}** "
                f"(для минимизатора это означает, что {COL_LABELS[col_to_remove]} нерационален)."
            )
    else:
        st.write("Для Игрока B доминируемых стратегий нет.")

    reduced_matrix, reduced_rows, reduced_cols, reduction_steps = reduce_by_dominance(
        PAYOFF_MATRIX, ROW_LABELS, COL_LABELS
    )

    if reduction_steps:
        st.write("После исключения доминируемых стратегий:")
        for step in reduction_steps:
            st.write(f"- {step}")
        st.dataframe(payoff_dataframe(reduced_matrix, reduced_rows, reduced_cols), use_container_width=True)

    st.subheader("3) Седловая точка (maxmin/minmax)")
    saddle = saddle_point_analysis(PAYOFF_MATRIX)
    row_mins = pd.DataFrame({"Стратегия A": ROW_LABELS, "min по строке": saddle["row_mins"]})
    col_maxs = pd.DataFrame({"Стратегия B": COL_LABELS, "max по столбцу": saddle["col_maxs"]})

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(row_mins, use_container_width=True)
    with c2:
        st.dataframe(col_maxs, use_container_width=True)

    st.write(f"Нижняя цена игры `maxmin = {saddle['lower_value']:.4f}`")
    st.write(f"Верхняя цена игры `minmax = {saddle['upper_value']:.4f}`")

    if saddle["has_saddle"]:
        st.success("Седловая точка есть, игра решается в чистых стратегиях.")
    else:
        st.warning("Седловой точки нет, нужны смешанные стратегии.")

    st.subheader("4) Графоаналитический метод (2xN)")
    graph_res = graphical_solution_2xN(
        matrix_original=PAYOFF_MATRIX,
        matrix_reduced=reduced_matrix,
        reduced_col_labels=reduced_cols,
        original_col_labels=COL_LABELS,
    )

    fig = go.Figure()
    for name, values in graph_res["lines"].items():
        style = dict(width=2)
        if name.startswith("B1"):
            style = dict(width=2, dash="dot")

        fig.add_trace(
            go.Scatter(
                x=graph_res["p_grid"],
                y=values,
                mode="lines",
                name=name,
                line=style,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=graph_res["p_grid"],
            y=graph_res["envelope"],
            mode="lines",
            name="Нижняя огибающая min_j g_j(p)",
            line=dict(width=4, color="black"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[graph_res["p_opt"]],
            y=[graph_res["value"]],
            mode="markers",
            name="Оптимум",
            marker=dict(size=11, color="red"),
        )
    )

    fig.add_vline(x=graph_res["p_opt"], line_dash="dash", line_color="red")
    fig.add_hline(y=graph_res["value"], line_dash="dash", line_color="red")

    fig.update_layout(
        title="Функции выигрыша A при p1 = P(A1)",
        xaxis_title="p1",
        yaxis_title="Ожидаемый выигрыш A",
        legend_title="Стратегии B",
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"Графоаналитическое решение: `p1={graph_res['p_opt']:.4f}`, `p2={graph_res['p2_opt']:.4f}`, "
        f"`V={graph_res['value']:.4f}`"
    )

    q_graph = graph_res["q_opt"]
    q_graph_text = ", ".join([f"q{i+1}={q_graph[i]:.4f}" for i in range(len(q_graph))])
    st.write(f"Оптимальная смешанная стратегия B (с учетом исходной 2x3): {q_graph_text}")

    st.subheader("5) Симплекс-метод (линейное программирование, scipy.optimize.linprog)")
    p_lp, v_lp_a, _ = simplex_solution_for_A(PAYOFF_MATRIX)
    q_lp, v_lp_b, _ = simplex_solution_for_B(PAYOFF_MATRIX)

    st.write(
        f"Решение ЛП для A: `p1={p_lp[0]:.4f}`, `p2={p_lp[1]:.4f}`, `V={v_lp_a:.4f}`"
    )
    st.write(
        f"Решение двойственной ЛП для B: "
        + ", ".join([f"q{i+1}={q_lp[i]:.4f}" for i in range(len(q_lp))])
        + f", `V={v_lp_b:.4f}`"
    )

    st.subheader("6) Метод Брауна–Робинсона")
    iterations = st.slider("Число итераций", min_value=100, max_value=5000, value=500, step=100)

    br_history, p_br, q_br = brown_robinson(PAYOFF_MATRIX, ROW_LABELS, COL_LABELS, iterations=iterations)

    st.write(
        f"После {iterations} итераций: `p1={p_br[0]:.4f}`, `p2={p_br[1]:.4f}`, "
        + ", ".join([f"q{i+1}={q_br[i]:.4f}" for i in range(len(q_br))])
    )
    st.write(
        f"Оценка цены игры по Брауну–Робинсону: `V≈{br_history['v_avg'].iloc[-1]:.4f}` "
        f"(интервал [{br_history['lower_v'].iloc[-1]:.4f}; {br_history['upper_v'].iloc[-1]:.4f}])"
    )

    # График сходимости
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=br_history["k"], y=br_history["lower_v"], mode="lines", name="Нижняя оценка"))
    fig_conv.add_trace(go.Scatter(x=br_history["k"], y=br_history["upper_v"], mode="lines", name="Верхняя оценка"))
    fig_conv.add_trace(go.Scatter(x=br_history["k"], y=br_history["v_avg"], mode="lines", name="Средняя оценка V"))
    fig_conv.add_hline(y=v_lp_a, line_dash="dash", line_color="green", annotation_text="V (linprog)")
    fig_conv.update_layout(
        title="Сходимость метода Брауна–Робинсона",
        xaxis_title="Итерация",
        yaxis_title="Оценка цены игры",
        height=420,
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    st.write("Таблица итераций (последние 25 шагов):")
    st.dataframe(br_history.tail(25), use_container_width=True)

    with st.expander("Показать полную таблицу итераций"):
        st.dataframe(br_history, use_container_width=True)

    st.subheader("7) Сравнение результатов")
    comparison = pd.DataFrame(
        [
            {
                "Метод": "Графоаналитический",
                "p1": graph_res["p_opt"],
                "p2": graph_res["p2_opt"],
                "q1": q_graph[0],
                "q2": q_graph[1],
                "q3": q_graph[2],
                "V": graph_res["value"],
            },
            {
                "Метод": "Симплекс (linprog)",
                "p1": p_lp[0],
                "p2": p_lp[1],
                "q1": q_lp[0],
                "q2": q_lp[1],
                "q3": q_lp[2],
                "V": v_lp_a,
            },
            {
                "Метод": f"Браун–Робинсон ({iterations} ит.)",
                "p1": p_br[0],
                "p2": p_br[1],
                "q1": q_br[0],
                "q2": q_br[1],
                "q3": q_br[2],
                "V": br_history["v_avg"].iloc[-1],
            },
        ]
    )

    st.dataframe(comparison.style.format({
        "p1": "{:.4f}",
        "p2": "{:.4f}",
        "q1": "{:.4f}",
        "q2": "{:.4f}",
        "q3": "{:.4f}",
        "V": "{:.4f}",
    }), use_container_width=True)

    st.info(
        "Пояснение для защиты: метод Брауна–Робинсона дает приближение. "
        "При увеличении числа итераций его оценка цены игры стремится к точному значению, "
        "полученному графоаналитическим и симплекс-методом."
    )


if __name__ == "__main__":
    main()
