import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli
from numpy import ndarray
from pandas.core.frame import DataFrame
from tqdm import tqdm

#from lab_01 import get_male_df, get_female_df, get_survival_rates, get_fertility_rate, get_newborn_girls_rate

feature_count = 21


def get_bounds_sur(pop_df: DataFrame) -> list:
    result = list()
    pop_lst = [row[1].to_numpy() for row in pop_df.iterrows()]

    for i in range(len(pop_lst) - 1):
        prev_range, new_range = pop_lst[i], pop_lst[i + 1]
        sur_rates = new_range[1:] / prev_range[:-1]

        if len(result) == 0:
            for rate in sur_rates:
                result.append([rate, rate])
        else:
            for j, rate in enumerate(sur_rates):
                result[j][0] = rate if result[j][0] > rate or pd.isna(result[j][0]) else result[j][0]
                result[j][1] = rate if result[j][1] < rate or pd.isna(result[j][1]) else result[j][1]

    return result


def get_bounds_fer(full_pop: DataFrame, f_pop: DataFrame) -> list:
    newborn_lst = [row[1].to_numpy()[0] for row in full_pop.iterrows()][1:]
    fem_pop_lst = [np.sum(row[1].to_numpy()[4:8]) for row in f_pop.iterrows()][:-1]
    fer_rates = [n / f for n, f in zip(newborn_lst, fem_pop_lst)]
    return [min(fer_rates), max(fer_rates)]


def get_bounds_ng(full_pop: DataFrame, f_pop: DataFrame) -> list:
    newborn_lst = [row[1].to_numpy()[0] for row in full_pop.iterrows()]
    newgirl_lst = [row[1].to_numpy()[0] for row in f_pop.iterrows()]
    newgirl_rates = [girl / both for both, girl in zip(newborn_lst, newgirl_lst)]
    return [min(newgirl_rates), max(newgirl_rates)]


def predict_population(in_male: ndarray, in_fem: ndarray, **kwargs) -> ndarray:
    predict_years = kwargs.get("predict_years", 10)
    year_step = 5
    iterations = predict_years // year_step

    m_sur_rates = kwargs.get("male_survival_rates")
    f_sur_rates = kwargs.get("female_survival_rates")
    fer_rate = kwargs.get("fertility_rate")
    girls_rate = kwargs.get("newborn_girls_rate")

    cur_m, prev_m = in_male[:feature_count].copy(), in_male[:feature_count].copy()
    cur_f, prev_f = in_fem[:feature_count].copy(), in_fem[:feature_count].copy()

    for i in range(year_step, iterations * year_step, year_step):
        newborn_cnt = fer_rate * np.sum(prev_f[4:8])

        prev_m, prev_f = cur_m, cur_f

        cur_m = np.insert(m_sur_rates * prev_m[:-1], 0, newborn_cnt * (1 - girls_rate))
        cur_f = np.insert(f_sur_rates * prev_f[:-1], 0, newborn_cnt * girls_rate)

    return np.sum(cur_m + cur_f)


def evaluate(params: ndarray, predict_years: int, m_df: DataFrame, f_df: DataFrame) -> ndarray:
    res_lst = list()
    input_male, input_female = m_df.loc[2005].to_numpy(), f_df.loc[2005].to_numpy()
    for param in tqdm(params):
        male_survival_rates = param[:feature_count - 1]
        female_survival_rates = param[feature_count - 1:(feature_count - 1) * 2]
        fertility_rate = param[-2]
        newborn_girls_rate = param[-1]

        res = predict_population(input_male, input_female,
                                 male_survival_rates=male_survival_rates, female_survival_rates=female_survival_rates,
                                 fertility_rate=fertility_rate, newborn_girls_rate=newborn_girls_rate,
                                 predict_years=predict_years)
        res_lst.append(res)

    return np.array(res_lst)


if __name__ == '__main__':
    both_df = pd.read_csv("both.csv", index_col="years")
    male_df = pd.read_csv("M.csv", index_col="years")
    fem_df = pd.read_csv("F.csv", index_col="years")

    problem = {
        "num_vars": (feature_count - 1) * 2 + 2,
        "names": [f"m_sur_rate_{i}" for i in range(feature_count - 1)] +
                 [f"f_sur_rate_{i}" for i in range(feature_count - 1)] +
                 ["fertility_rate", "newborn_girls_rate"],
        "bounds": [
            *get_bounds_sur(male_df)[:feature_count-1], *get_bounds_sur(fem_df)[:feature_count-1],
            get_bounds_fer(both_df, fem_df), get_bounds_ng(both_df, fem_df)
        ]
    }
    n_samples = 16
    param_values = saltelli.sample(problem, n_samples)

    result_si_list, result_y_list = [], []
    time_predictions = list(range(10, 20, 10))
    for time_pred in time_predictions:
        # Compute results
        y = evaluate(param_values, time_pred, male_df, fem_df)
        si = sobol.analyze(problem, y, print_to_console=False)

        si_res = np.array([
            np.sum(si["S1"][:feature_count - 1]),
            np.sum(si["S1"][feature_count - 1:(feature_count - 1) * 2]),
            si["S1"][-2], si["S1"][-1]
        ])

        # Save results
        with open(f"time_pred_{time_pred}.npy", "wb") as f:
            np.save(f, y)
        with open(f"sobol_{time_pred}.npy", "wb") as f:
            np.save(f, si_res)

        # # Load results
        # with open(f"data/time_pred_{time_pred}.npy", "rb") as f:
        #     y = np.load(f)
        # with open(f"data/sobol_{time_pred}.npy", "rb") as f:
        #     si_res = np.load(f)

        result_y_list.append(y)
        result_si_list.append(si_res)

        print("=" * 80)
        print(f"{time_pred:3d} years: " + str(si_res))

    fig, (ax_sens, ax_pop) = plt.subplots(2, figsize=(10, 8), sharex=True)
    x_axis = np.array(time_predictions)

    m_sur_sens = np.array([res[0] for res in result_si_list])
    f_sur_sens = np.array([res[1] for res in result_si_list])
    fer_f_sens = np.array([res[2] for res in result_si_list])
    new_g_sens = np.array([res[3] for res in result_si_list])

    print("=" * 80)
    print("Sensitivity analysis")
    print(f"male survival rate:    \n{m_sur_sens}")
    print(f"female survival rate:  \n{f_sur_sens}")
    print(f"fertility rate rate:   \n{fer_f_sens}")
    print(f"newborn girl :\n{new_g_sens}")

    ax_sens.set_title("Sensitivity analysis")
    ax_sens.scatter(x_axis, m_sur_sens, label="male survival")
    ax_sens.scatter(x_axis, f_sur_sens, label="female survival")
    ax_sens.scatter(x_axis, fer_f_sens, label="fertility rate")
    ax_sens.scatter(x_axis, new_g_sens, label="newborn girl ")
    ax_sens.legend()
    ax_sens.grid()

    print("=" * 80)
    print("Uncertainty analysis")

    ax_pop.set_title("Uncertainty analysis")
    max_plot = 100
    plot_results = [np.random.choice(res_y, max_plot, replace=False) for res_y in result_y_list]
    for i in tqdm(range(max_plot)):
        ax_pop.scatter(x_axis, [res_y[i] for res_y in plot_results])
    ax_pop.fill_between(
        x_axis,
        [np.percentile(r, 2.5) for r in result_y_list],
        [np.percentile(r, 97.5) for r in result_y_list],
        alpha=0.5
    )
    ax_pop.grid()

    fig.canvas.manager.set_window_title("Lab_02")
    plt.show()
