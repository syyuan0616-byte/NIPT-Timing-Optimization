import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({'font.size': 20})

df = pd.read_csv('forlasso.csv')
X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练带L1正则化的逻辑回归（LASSO分类）
lasso_clf = LogisticRegressionCV(
    Cs=np.logspace(-4, 2, 100),  # 测试100个正则化参数
    penalty='l1',                # L1正则化（LASSO）
    solver='saga',               # 支持L1正则化的求解器
    cv=5,                        # 5折交叉验证
    random_state=42,
    max_iter=10000,              # 确保迭代收敛
    scoring='accuracy'
)

lasso_clf.fit(X_train_scaled, y_train)

print(f"\n最优正则化参数C：{lasso_clf.C_[0]:.4f}")

# 提取因素筛选结果
coef_df = pd.DataFrame({
    '因素': df.columns[:13],
    '系数': lasso_clf.coef_[0],
    '影响方向': ['增加不健康风险' if coef > 0 else '降低不健康风险' for coef in lasso_clf.coef_[0]]
})

# 筛选出非零系数
selected_factors = coef_df[coef_df['系数'] != 0].sort_values(by='系数', key=abs, ascending=False)

print(f"在13个因素中，LASSO筛选出{len(selected_factors)}个显著影响因素：")
print(selected_factors)

# 模型评估（针对健康/不健康分类）
y_pred = lasso_clf.predict(X_test_scaled)

print(f"测试集准确率：{accuracy_score(y_test, y_pred):.4f}（正确分类的样本比例）")
print("\n混淆矩阵")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index=['实际健康（0）', '实际不健康（1）'],
                     columns=['预测健康（0）', '预测不健康（1）'])
print(cm_df)
print("\n分类报告（精确率、召回率、F1值）：")
print(classification_report(y_test, y_pred, target_names=['健康（0）', '不健康（1）']))

# 交叉验证结果图
plt.figure(figsize=(10, 6))
mean_scores = np.mean(lasso_clf.scores_[1], axis=0)
plt.plot(np.log10(lasso_clf.Cs_), mean_scores, 'b-', linewidth=2)

# 标记最优参数位置
plt.axvline(x=np.log10(lasso_clf.C_[0]), color='r', linestyle='--',
            label=f'最优C值: log10(C)={np.log10(lasso_clf.C_[0]):.2f}')

plt.xlabel('log10(C)')
plt.ylabel('交叉验证准确率')
plt.title('不同正则化参数下的LASSO交叉验证结果')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# LASSO系数路径图（展示因素对健康状态的影响变化）
Cs = np.logspace(-4, 2, 100)
coefs = []

for c in Cs:
    clf = LogisticRegression(
        C=c,
        penalty='l1',
        solver='saga',
        random_state=42,
        max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    coefs.append(clf.coef_[0])

coefs = np.array(coefs)

plt.figure(figsize=(12, 8))
for i in range(coefs.shape[1]):
    plt.plot(np.log10(Cs), coefs[:, i], label=df.columns[i])

plt.axvline(x=np.log10(lasso_clf.C_[0]), color='r', linestyle='--',
            label=f'最优C值: log10(C)={np.log10(lasso_clf.C_[0]):.2f}')

plt.xlabel('log10(C)')
plt.ylabel('系数值（正值=增加不健康风险）')
plt.title('LASSO系数路径图（系数随正则化强度的变化）')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放右侧
plt.tight_layout()
plt.show()

# 重要因素对健康状态的影响强度
plt.figure(figsize=(10, 6))
bars = plt.bar(selected_factors['因素'], selected_factors['系数'])

# 为柱状图添加颜色
for bar, coef in zip(bars, selected_factors['系数']):
    if coef > 0:
        bar.set_color('salmon')  # 增加不健康风险的因素标为浅红
    else:
        bar.set_color('lightgreen')  # 降低不健康风险的因素标为浅绿

plt.axhline(y=0, color='r', linestyle='--')
plt.title('影响健康状态的重要因素及其强度')
plt.xticks(rotation=45, ha='right')  # 标签右对齐
plt.ylabel('系数值（绝对值越大影响越强）')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(lasso_clf.intercept_)
