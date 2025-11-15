import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_data(file_path):
    """加载Excel数据并返回处理后的DataFrame"""
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载数据，共 {len(df)} 条记录")
        
        # 检查必要列
        required_cols = ['序号', '孕妇代码', '年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
            
        # 处理缺失值
        df = df.dropna(subset=required_cols)
        
        # 处理异常值（基于Y染色体浓度和BMI）
        for col in ['孕妇BMI', 'Y染色体浓度']:
            q1, q3 = df[col].quantile([0.01, 0.99])
            df = df[(df[col] >= q1) & (df[col] <= q3)]
            
        print(f"处理后数据量: {len(df)} 条 (已移除异常值)")
        return df
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None

def calculate_weights_based_on_linear_regression(df):
    """通过多元线性回归计算各参数对Y染色体浓度的影响权重"""
    # 定义自变量和因变量
    X = df[['年龄', '身高', '体重', '孕妇BMI']]
    y = df['Y染色体浓度']
    
    # 标准化自变量
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 拟合多元线性回归模型
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 回归系数（表示各因素对Y染色体浓度的影响）
    coefficients = model.coef_
    
    # 将系数转换为权重（取绝对值并归一化）
    weights = np.abs(coefficients)
    weights = weights / np.sum(weights)  # 归一化，使权重之和为1
    
    # 显示回归结果
    print("\n===== 多元线性回归结果 =====")
    print(f"回归方程: Y染色体浓度 = {model.intercept_:.4f} + " + 
          " + ".join([f"{coef:.4f}×{col}" for coef, col in zip(model.coef_, X.columns)]))
    print(f"模型R²值: {model.score(X_scaled, y):.4f} (越接近1表示拟合越好)")
    
    # 显示权重
    print("\n===== 基于回归系数的权重 =====")
    for col, weight in zip(X.columns, weights):
        print(f"{col}: {weight:.4f} ({weight*100:.2f}%)")
    
    return weights

def eight_clusters_bmi_clustering(df, weights):
    """固定8类的聚类分析"""
    # 选择用于聚类的特征
    features = ['年龄', '身高', '体重', '孕妇BMI']
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # 应用权重
    weighted_features = scaled_features * weights
    
    # 执行K-means聚类（固定8类）
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # 增加n_init确保稳定性
    df['聚类分组'] = kmeans.fit_predict(weighted_features)
    
    # 按BMI均值排序分组（确保分组按BMI从低到高排列）
    bmi_means = df.groupby('聚类分组')['孕妇BMI'].mean().sort_values()
    group_mapping = {old_group: new_group for new_group, old_group in enumerate(bmi_means.index)}
    df['聚类分组'] = df['聚类分组'].map(group_mapping)
    
    # 计算聚类效果指标
    silhouette_avg = silhouette_score(weighted_features, df['聚类分组'])
    print(f"\n8类聚类轮廓系数: {silhouette_avg:.4f} (越接近1效果越好)")
    
    return df

def analyze_and_output_eight_groups(df, save_path):
    """分析并输出8类聚类的结果"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 按聚类分组统计详细信息
    groups_stats = df.groupby('聚类分组').agg({
        '孕妇代码': 'count',
        '孕妇BMI': ['mean', 'std', 'min', 'max'],
        'Y染色体浓度': ['mean', 'std', 'min', 'max'],
        '年龄': ['mean', 'std'],
        '身高': ['mean', 'std'],
        '体重': ['mean', 'std']
    }).round(2)
    
    # 重命名列
    groups_stats.columns = [
        '样本数量', 'BMI均值', 'BMI标准差', 'BMI最小值', 'BMI最大值',
        'Y浓度均值', 'Y浓度标准差', 'Y浓度最小值', 'Y浓度最大值',
        '平均年龄', '年龄标准差', '平均身高', '身高标准差',
        '平均体重', '体重标准差'
    ]
    
    # 显示8类分组结果
    print("\n===== 8类BMI聚类分组结果 =====")
    print(groups_stats)
    
    # 保存分组统计结果
    stats_path = os.path.join(save_path, '8_clusters_statistics.xlsx')
    groups_stats.to_excel(stats_path)
    print(f"\n已保存8类分组统计结果至: {stats_path}")
    
    # 生成可视化图表
    plt.figure(figsize=(18, 15))
    
    # 1. BMI分布箱线图
    plt.subplot(3, 2, 1)
    sns.boxplot(x='聚类分组', y='孕妇BMI', data=df, palette='Set2')
    plt.title('8类分组的BMI分布', fontsize=14)
    plt.xlabel('聚类分组 (按BMI从低到高)', fontsize=12)
    plt.ylabel('BMI值', fontsize=12)
    
    # 2. Y染色体浓度分布
    plt.subplot(3, 2, 2)
    sns.boxplot(x='聚类分组', y='Y染色体浓度', data=df, palette='Set2')
    plt.title('8类分组的Y染色体浓度分布', fontsize=14)
    plt.xlabel('聚类分组', fontsize=12)
    plt.ylabel('Y染色体浓度', fontsize=12)
    
    # 3. BMI与Y染色体浓度的关系
    plt.subplot(3, 2, 3)
    sns.scatterplot(x='孕妇BMI', y='Y染色体浓度', hue='聚类分组', data=df, 
                   palette='tab10', s=60, alpha=0.7)
    plt.title('BMI与Y染色体浓度的关系', fontsize=14)
    plt.xlabel('BMI值', fontsize=12)
    plt.ylabel('Y染色体浓度', fontsize=12)
    plt.legend(title='聚类分组', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. 各分组样本数量
    plt.subplot(3, 2, 4)
    sns.countplot(x='聚类分组', data=df, palette='Set2')
    plt.title('各分组样本数量分布', fontsize=14)
    plt.xlabel('聚类分组', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    
    # 5. 各因素权重可视化
    features = ['年龄', '身高', '体重', '孕妇BMI']
    weights = calculate_weights_based_on_linear_regression(df)
    plt.subplot(3, 2, 5)
    plt.bar(features, weights)
    plt.title('各因素对Y染色体浓度的影响权重', fontsize=14)
    plt.xlabel('因素', fontsize=12)
    plt.ylabel('权重值', fontsize=12)
    for i, v in enumerate(weights):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    viz_path = os.path.join(save_path, '8_clusters_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"已保存8类分组可视化图表至: {viz_path}")
    
    # 保存包含聚类结果的完整数据
    result_df = df.sort_values(['聚类分组', '孕妇BMI'])[
        ['序号', '孕妇代码', '聚类分组', '孕妇BMI', 'Y染色体浓度', 
         '年龄', '身高', '体重']
    ]
    result_path = os.path.join(save_path, '8_clusters_results.xlsx')
    result_df.to_excel(result_path, index=False)
    print(f"已保存包含8类分组的完整数据至: {result_path}")
    
    return groups_stats

def main():
    # 配置参数
    file_path = "附件.xlsx"  # 输入你的Excel文件路径
    
    # 设置保存路径为桌面
    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "8类BMI聚类结果")
    print(f"结果将保存至: {save_path}")
    
    # 加载数据
    df = load_data(file_path)
    if df is None:
        return
    
    # 检查数据量是否适合8类聚类
    if len(df) < 40:  # 每个类别至少需要一定样本量
        print(f"警告: 样本量较小({len(df)})，8类聚类可能效果不佳")
    
    # 基于多元线性回归计算权重
    weights = calculate_weights_based_on_linear_regression(df)
    
    # 执行8类聚类
    df = eight_clusters_bmi_clustering(df, weights)
    
    # 分析并输出结果
    analyze_and_output_eight_groups(df, save_path)
    
    # 打印分组区间总结
    print("\n===== 8类BMI分组区间总结 =====")
    for group in sorted(df['聚类分组'].unique()):
        group_data = df[df['聚类分组'] == group]
        bmi_min = group_data['孕妇BMI'].min()
        bmi_max = group_data['孕妇BMI'].max()
        y_mean = group_data['Y染色体浓度'].mean()
        print(f"分组 {group}: BMI [{bmi_min:.1f}, {bmi_max:.1f}], 平均Y浓度: {y_mean:.2f}, 样本数: {len(group_data)}")

if __name__ == "__main__":
    main()
