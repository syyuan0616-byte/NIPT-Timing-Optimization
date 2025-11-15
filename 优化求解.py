import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.stats import weibull_min, lognorm

# --------------------------
# 1. 参数定义（保持不变）
# --------------------------
new_interval_params = [
    {"age_mean": 28.71875, "height_mean": 160.875, "weight_mean": 74.3853125},
    {"age_mean": 29.08823529, "height_mean": 160.5661765, "weight_mean": 78.49764706},
    {"age_mean": 28.75757576, "height_mean": 161.6060606, "weight_mean": 84.37515152},
    {"age_mean": 29.16901408, "height_mean": 161.443662, "weight_mean": 93.06098592}
]

bmi_intervals_3 = [
    {
        "bmi_range": [20.0, 29.75],
        "bmi_mean": 24.875,
        "age_mean": new_interval_params[0]["age_mean"],
        "height_mean": new_interval_params[0]["height_mean"],
        "weight_mean": new_interval_params[0]["weight_mean"],
        "D": 1,
        "mu": 1,
        "dist_type": "weibull",
        "params": {"k": 3.9234, "lambda": 21.7806, "gamma": 67.7014}
    },
    {
        "bmi_range": [29.75, 31.25],
        "bmi_mean": 30.5,
        "age_mean": new_interval_params[1]["age_mean"],
        "height_mean": new_interval_params[1]["height_mean"],
        "weight_mean": new_interval_params[1]["weight_mean"],
        "D": 1,
        "mu": 1,
        "dist_type": "weibull",
        "params": {"k": 3.7681, "lambda": 17.8685, "gamma": 71.7896}
    },
    {
        "bmi_range": [31.25, 33.33],
        "bmi_mean": 32.29,
        "age_mean": new_interval_params[2]["age_mean"],
        "height_mean": new_interval_params[2]["height_mean"],
        "weight_mean": new_interval_params[2]["weight_mean"],
        "D": 1,
        "mu": 1,
        "dist_type": "lognorm",
        "params": {"mu": 3.6593, "sigma": 0.1560}
    },
    {
        "bmi_range": [33.33, 48.0],
        "bmi_mean": 35,
        "age_mean": new_interval_params[3]["age_mean"],
        "height_mean": new_interval_params[3]["height_mean"],
        "weight_mean": new_interval_params[3]["weight_mean"],
        "D": 1,
        "mu": 1,
        "dist_type": "weibull",
        "params": {"k": 4.0383, "lambda": 19.7330, "gamma": 70.8231}
    }
]

CONCENTRATION_THRESHOLD = 0.04
PROPORTION_THRESHOLD = 0.86


# --------------------------
# 2. 严重度与频率计算（保持不变）
# --------------------------
def calc_P(t):
    if 0 <= t <= 84:
        return 0.05 * np.exp(0.017 * t)
    elif 85 <= t <= 189:
        P84 = 0.05 * np.exp(0.017 * 84)
        return P84 * np.exp(0.011 * (t - 84))
    else:
        P84 = 0.05 * np.exp(0.017 * 84)
        P189 = P84 * np.exp(0.011 * (189 - 84))
        return P189 * np.exp(0.014 * (t - 189))


def calc_I(t):
    if 0 <= t <= 84:
        return 0.04 * np.exp(0.018 * t)
    elif 85 <= t <= 189:
        I84 = 0.04 * np.exp(0.018 * 84)
        return I84 * np.exp(0.013 * (t - 84))
    else:
        I84 = 0.04 * np.exp(0.018 * 84)
        I189 = I84 * np.exp(0.013 * (189 - 84))
        return I189 * np.exp(0.015 * (t - 189))


P0I0 = calc_P(0) * calc_I(0)
P280I280 = calc_P(280) * calc_I(280)


def calc_S(t):
    PI = calc_P(t) * calc_I(t)
    S_raw = (PI - P0I0) / (P280I280 - P0I0) * 10
    return np.minimum(10.0, np.maximum(0.0, S_raw))


def calc_O(t, interval):
    dist_type = interval["dist_type"]
    params = interval["params"]
    mu = interval["mu"]

    if t <= 0:
        return 0.0

    if dist_type == "weibull":
        k = params["k"]
        lam = params["lambda"]
        gamma = params["gamma"]
        p_t = weibull_min.pdf(t - gamma, k, scale=lam) if t >= gamma else 0.0
    elif dist_type == "lognorm":
        sigma = params["sigma"]
        mu_param = params["mu"]
        p_t = lognorm.pdf(t, s=sigma, scale=np.exp(mu_param))
    else:
        p_t = 0.0

    return mu * p_t + (1 - mu)


# --------------------------
# 3. 目标函数与约束（保持不变）
# --------------------------
def objective(t_list):
    rpn_total = 0.0
    for i in range(4):
        t = t_list[i]
        interval = bmi_intervals_3[i]
        S = calc_S(t)
        O = calc_O(t, interval)
        D = interval["D"]
        rpn_total += S * O * D
    return rpn_total


def con_concentration_1(t):
    return -0.001 * t[0] + 0.0086 * 24.875 - 0.0002 * (24.875 ** 2) + 0.0776


def con_concentration_2(t):
    return -0.001 * t[1] + 0.0086 * 30.5 - 0.0002 * (30.5 ** 2) + 0.0776


def con_concentration_3(t):
    return -0.001 * t[2] + 0.0086 * 32.29 - 0.0002 * (32.29 ** 2) + 0.0776


def con_concentration_4(t):
    return -0.001 * t[3] + 0.0086 * 35 - 0.0002 * (35 ** 2) + 0.0776


def con_proportion_1(t):
    return 3.2282 - 0.0044 * 28.71875 - 0.0142 * 160.875 + 0.0004 * 74.3853125 + 0.0002 * t[0]


def con_proportion_2(t):
    return 3.2282 - 0.0044 * 29.08823529 - 0.0142 * 160.5661765 + 0.0004 * 78.49764706 + 0.0002 * t[1]


def con_proportion_3(t):
    return 3.2282 - 0.0044 * 28.75757576 - 0.0142 * 161.6060606 + 0.0004 * 84.37515152 + 0.0002 * t[2]


def con_proportion_4(t):
    return 3.2282 - 0.0044 * 29.16901408 - 0.0142 * 161.443662 + 0.0004 * 93.06098592 + 0.0002 * t[3]


constraints = [
    NonlinearConstraint(con_concentration_1, CONCENTRATION_THRESHOLD, np.inf),
    NonlinearConstraint(con_proportion_1, PROPORTION_THRESHOLD, np.inf),
    NonlinearConstraint(con_concentration_2, CONCENTRATION_THRESHOLD, np.inf),
    NonlinearConstraint(con_proportion_2, PROPORTION_THRESHOLD, np.inf),
    NonlinearConstraint(con_concentration_3, CONCENTRATION_THRESHOLD, np.inf),
    NonlinearConstraint(con_proportion_3, PROPORTION_THRESHOLD, np.inf),
    NonlinearConstraint(con_concentration_4, CONCENTRATION_THRESHOLD, np.inf),
    NonlinearConstraint(con_proportion_4, PROPORTION_THRESHOLD, np.inf)
]

bounds = [(77.0, 203.0) for _ in range(4)]


# --------------------------
# 4. 约束验证函数（保持不变）
# --------------------------
def concentration_constraint(t, interval):
    bmi_mean = interval["bmi_mean"]
    return -0.001 * t + 0.0086 * bmi_mean - 0.0002 * (bmi_mean ** 2) + 0.0776


def proportion_constraint(t, interval):
    a = interval["age_mean"]
    h = interval["height_mean"]
    w = interval["weight_mean"]
    return 3.2282 - 0.0044 * a - 0.0142 * h + 0.0004 * w + 0.0002 * t


# --------------------------
# 5. 求解部分（修复梯度警告）
# --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("问题三优化模型求解（消除梯度警告）")
    print("=" * 60)

    # 约束可行性验证
    print("约束可行性预验证:")
    for i, interval in enumerate(bmi_intervals_3):
        conc_min = concentration_constraint(77.0, interval)
        conc_max = concentration_constraint(203.0, interval)
        prop_min = proportion_constraint(77.0, interval)
        prop_max = proportion_constraint(203.0, interval)

        print(f"区间{i + 1} [BMI {interval['bmi_range'][0]:.2f}-{interval['bmi_range'][1]:.2f}]:")
        print(f"  浓度约束范围: {conc_min:.4f} ~ {conc_max:.4f}")
        print(f"  比例约束范围: {prop_min:.4f} ~ {prop_max:.4f}\n")

    # 修复：禁用polish步骤以避免拟牛顿方法引发的警告
    print("开始优化计算...")
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        constraints=constraints,
        strategy='best1bin',
        maxiter=1000,
        popsize=20,
        tol=1e-5,
        mutation=0.8,
        recombination=0.7,
        polish=False,  # 禁用局部精细优化，消除梯度警告
        disp=True
    )

    # 输出结果（保持不变）
    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)

    if result.success:
        optimal_t = result.x
        min_rpn = result.fun

        print(f"优化状态: 成功")
        print(f"最小总RPN值: {min_rpn:.6f}")
        print(f"迭代次数: {result.nit}")
        print(f"函数评估次数: {result.nfev}\n")

        print("各BMI区间最优NIPT时点:")
        for i in range(4):
            interval = bmi_intervals_3[i]
            bmi_range = f"[{interval['bmi_range'][0]:.2f}, {interval['bmi_range'][1]:.2f})"
            print(f"区间{i + 1} {bmi_range}: {optimal_t[i]:.2f}天 (约{optimal_t[i] / 7:.1f}周)")

        print(f"\n约束验证:")
        for i in range(4):
            t = optimal_t[i]
            interval = bmi_intervals_3[i]
            conc_val = concentration_constraint(t, interval)
            prop_val = proportion_constraint(t, interval)
            print(f"区间{i + 1}: 浓度={conc_val:.4f}, 比例={prop_val:.4f}")
    else:
        print(f"优化失败: {result.message}")

    print("=" * 60)
