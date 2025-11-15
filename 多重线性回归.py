import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
import re

# 设置中文显示并确保负号正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')  # 忽略警告信息

# 1. 辅助函数：提取字符串中的数字
def extract_number(value):
    """从字符串中提取数字，处理'≥3'、'>2'、'3+'等格式"""
    try:
        if pd.api.types.is_numeric_dtype(type(value)):
            return float(value)
            
        value_str = str(value).strip()
        if not value_str:
            return np.nan
            
        match = re.search(r'(\d+\.?\d*)', value_str)
        if match:
            return float(match.group(1))
        else:
            return np.nan
    except:
        return np.nan

# 2. 转换孕周格式（如'13w+6'转为13.857）
def convert_gestational_age(age_str):
    """将孕周字符串转换为数值"""
    try:
        if pd.api.types.is_numeric_dtype(type(age_str)):
            return float(age_str)
            
        age_str = str(age_str).strip().lower()
        if 'w' in age_str:
            week_part, day_part = age_str.split('w')
            week = float(week_part)
            
            if '+' in day_part:
                day = float(day_part.split('+')[1])
            elif day_part.strip():
                day = float(day_part)
            else:
                day = 0
                
            return week + (day / 7)
        else:
            return extract_number(age_str)
    except:
        return np.nan

# 3. 处理妊娠类型（自然受孕、IUI、IVF）
def convert_pregnancy_type(type_str):
    """
    将妊娠类型转换为数值权重：
    - 自然受孕：0（无干预）
    - IUI（人工授精）：1（轻度干预）
    - IVF（试管婴儿）：2（重度干预）
    """
    try:
        if pd.api.types.is_numeric_dtype(type_str):
            return float(type_str)
            
        type_str = str(type_str).strip().lower()
        
        if '自然' in type_str or '正常' in type_str:
            return 0
        elif 'iui' in type_str or '人工授精' in type_str:
            return 1
        elif 'ivf' in type_str or '试管婴儿' in type_str:
            return 2
        else:
            print(f"警告：无法识别的妊娠类型 '{type_str}'")
            return np.nan
    except:
        return np.nan

# 4. 数据加载
def load_data(file_path):
    """加载Excel数据文件"""
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载Excel数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
        print("数据列名：", df.columns.tolist())
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 5. 数据预处理（不筛选GC含量，使用全部数据）
def preprocess_data(df):
    """数据预处理：处理特殊值、不筛选GC含量、处理缺失值"""
    # 使用全部数据，不根据GC含量筛选
    df_processed = df.copy()
    print(f"使用全部数据进行分析，共 {df_processed.shape[0]} 行")
    
    # 处理GC含量（如果存在）
    if "GC含量" in df_processed.columns:
        print("正在处理GC含量数据...")
        df_processed["GC含量"] = df_processed["GC含量"].apply(extract_number)
        
        # 处理GC含量的缺失值
        gc_missing = df_processed["GC含量"].isnull().sum()
        if gc_missing > 0:
            print(f"注意：GC含量列有 {gc_missing} 个缺失值，将用中位数填充")
            df_processed["GC含量"].fillna(df_processed["GC含量"].median(), inplace=True)
        
        # 显示GC含量的分布统计
        print("GC含量分布统计:")
        print(df_processed["GC含量"].describe())
    
    # 处理其他需要转换为数值的列
    numeric_columns = ["年龄", "身高", "体重", "检测孕周", "孕妇BMI", 
                      "Y染色体浓度", "怀孕次数", "生产次数", "检测抽血次数"]
    
    for col in numeric_columns:
        if col in df_processed.columns:
            print(f"正在处理'{col}'列的特殊值...")
            if col == "检测孕周":
                df_processed[col] = df_processed[col].apply(convert_gestational_age)
            else:
                df_processed[col] = df_processed[col].apply(extract_number)
            
            invalid_count = df_processed[col].isnull().sum()
            if invalid_count > 0:
                print(f"注意：'{col}'列有 {invalid_count} 个值无法转换为数字，将用中位数填充")
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # 处理妊娠类型（自然受孕、IUI、IVF）
    if "IVF妊娠" in df_processed.columns:
        print("正在处理妊娠类型数据...")
        df_processed = df_processed.rename(columns={"IVF妊娠": "妊娠类型"})
        df_processed["妊娠类型"] = df_processed["妊娠类型"].apply(convert_pregnancy_type)
        
        # 统计各类别的数量
        type_counts = df_processed["妊娠类型"].value_counts().to_dict()
        print(f"妊娠类型分布：")
        print(f"  自然受孕: {type_counts.get(0, 0)}")
        print(f"  IUI(人工授精): {type_counts.get(1, 0)}")
        print(f"  IVF(试管婴儿): {type_counts.get(2, 0)}")
        
        # 处理无法识别的类型
        invalid_count = df_processed["妊娠类型"].isnull().sum()
        if invalid_count > 0:
            print(f"注意：有 {invalid_count} 个妊娠类型无法识别，将用最常见类型填充")
            most_common = df_processed["妊娠类型"].mode()[0] if not df_processed["妊娠类型"].mode().empty else 0
            df_processed["妊娠类型"].fillna(most_common, inplace=True)
    
    # 处理Y染色体浓度异常值
    if "Y染色体浓度" in df_processed.columns:
        z_scores = np.abs(stats.zscore(df_processed["Y染色体浓度"]))
        df_processed = df_processed[(z_scores < 3)]
        print(f"处理Y染色体浓度异常值后的样本量: {df_processed.shape[0]} 行")
    
    return df_processed

# 6. 探索性数据分析（增加GC含量相关分析）
def exploratory_analysis(df):
    """探索性数据分析，包含GC含量分析"""
    print("\n基本统计描述:")
    stats_vars = ["年龄", "孕妇BMI", "检测孕周", "Y染色体浓度", "妊娠类型"]
    if "GC含量" in df.columns:
        stats_vars.insert(0, "GC含量")  # 将GC含量加入统计变量
    stats_vars = [var for var in stats_vars if var in df.columns]
    print(df[stats_vars].describe())
    
    # 相关性分析（包含GC含量）
    plt.figure(figsize=(12, 10))
    corr_vars = ["年龄", "身高", "体重", "孕妇BMI", "检测孕周", 
                "Y染色体浓度", "怀孕次数", "生产次数", "检测抽血次数", "妊娠类型"]
    if "GC含量" in df.columns:
        corr_vars.insert(0, "GC含量")  # 将GC含量加入相关性分析
    
    corr_vars = [var for var in corr_vars if var in df.columns]
    
    # 确保所有用于相关性分析的变量都是数值型
    valid_corr_vars = []
    for var in corr_vars:
        if pd.api.types.is_numeric_dtype(df[var].dtype):
            valid_corr_vars.append(var)
        else:
            print(f"警告：'{var}'列不是数值型，已从相关性分析中排除")
    
    if len(valid_corr_vars) >= 2:
        corr_matrix = df[valid_corr_vars].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("变量相关性热力图（包含GC含量）")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=300)
        plt.close()
        print("相关性热力图已保存为 correlation_heatmap.png")
    else:
        print("有效数值变量不足，无法绘制相关性热力图")
    
    # 增加GC含量与Y染色体浓度的关系图
    if "Y染色体浓度" in df.columns and "GC含量" in df.columns:
        plt.figure(figsize=(15, 10))
        
        # GC含量与Y染色体浓度的关系
        plt.subplot(221)
        sns.regplot(x="GC含量", y="Y染色体浓度", data=df,
                   scatter_kws={"alpha":0.3}, line_kws={"color":"green"})
        plt.title("Y染色体浓度与GC含量的关系")
        
        # 孕周与Y染色体浓度的关系
        if "检测孕周" in df.columns and pd.api.types.is_numeric_dtype(df["检测孕周"].dtype):
            plt.subplot(222)
            hue_col = "妊娠类型" if "妊娠类型" in df.columns else None
            sns.scatterplot(x="检测孕周", y="Y染色体浓度", hue=hue_col, data=df)
            plt.title("Y染色体浓度与孕周的关系")
            if hue_col:
                plt.legend(title="妊娠类型", labels=["自然受孕", "IUI", "IVF"])
        
        # BMI与Y染色体浓度的关系
        if "孕妇BMI" in df.columns and pd.api.types.is_numeric_dtype(df["孕妇BMI"].dtype):
            plt.subplot(223)
            hue_col = "妊娠类型" if "妊娠类型" in df.columns else None
            sns.scatterplot(x="孕妇BMI", y="Y染色体浓度", hue=hue_col, data=df)
            plt.title("Y染色体浓度与BMI的关系")
            if hue_col:
                plt.legend(title="妊娠类型", labels=["自然受孕", "IUI", "IVF"])
        
        # 妊娠类型与Y染色体浓度的关系
        if "妊娠类型" in df.columns and pd.api.types.is_numeric_dtype(df["妊娠类型"].dtype):
            plt.subplot(224)
            sns.boxplot(x="妊娠类型", y="Y染色体浓度", data=df)
            plt.title("不同妊娠类型的Y染色体浓度分布")
            plt.xticks([0, 1, 2], ["自然受孕", "IUI", "IVF"])
        
        plt.tight_layout()
        plt.savefig("y_chromosome_relationships.png", dpi=300)
        plt.close()
        print("关系图已保存为 y_chromosome_relationships.png")
    else:
        # 如果没有GC含量数据，使用原来的关系图
        if "Y染色体浓度" in df.columns:
            plt.figure(figsize=(15, 10))
            plot_idx = 1
            
            if "检测孕周" in df.columns:
                plt.subplot(221)
                hue_col = "妊娠类型" if "妊娠类型" in df.columns else None
                sns.scatterplot(x="检测孕周", y="Y染色体浓度", hue=hue_col, data=df)
                plt.title("Y染色体浓度与孕周的关系")
                plot_idx += 1
            
            if "孕妇BMI" in df.columns:
                plt.subplot(222)
                hue_col = "妊娠类型" if "妊娠类型" in df.columns else None
                sns.scatterplot(x="孕妇BMI", y="Y染色体浓度", hue=hue_col, data=df)
                plt.title("Y染色体浓度与BMI的关系")
                plot_idx += 1
            
            if "检测抽血次数" in df.columns:
                plt.subplot(223)
                sns.boxplot(x="检测抽血次数", y="Y染色体浓度", data=df)
                plt.title("不同抽血次数的Y染色体浓度分布")
                plot_idx += 1
            
            if "妊娠类型" in df.columns:
                plt.subplot(224)
                sns.boxplot(x="妊娠类型", y="Y染色体浓度", data=df)
                plt.title("不同妊娠类型的Y染色体浓度分布")
                plt.xticks([0, 1, 2], ["自然受孕", "IUI", "IVF"])
            
            plt.tight_layout()
            plt.savefig("y_chromosome_relationships.png", dpi=300)
            plt.close()
            print("关系图已保存为 y_chromosome_relationships.png")

# 7. 构建回归模型（包含GC含量作为自变量）
def build_regression_model(df):
    """构建多变量回归模型，包含GC含量（如果存在）"""
    if "Y染色体浓度" not in df.columns:
        print("错误：数据中没有'Y染色体浓度'列，无法进行回归分析")
        return None, []
    
    # 选择自变量，包含GC含量（如果存在）
    candidate_vars = ["年龄", "孕妇BMI", "检测孕周", "妊娠类型", 
                      "检测抽血次数", "怀孕次数", "生产次数"]
    if "GC含量" in df.columns:
        candidate_vars.insert(0, "GC含量")  # 将GC含量作为自变量之一
    
    independent_vars = []
    
    for var in candidate_vars:
        if var in df.columns and pd.api.types.is_numeric_dtype(df[var].dtype):
            independent_vars.append(var)
        elif var in df.columns:
            print(f"警告：'{var}'列不是数值型，已从回归模型中排除")
    
    if not independent_vars:
        print("没有找到有效的自变量，无法构建模型")
        return None, []
    
    print(f"\n将使用以下自变量进行回归分析: {independent_vars}")
    
    # 检查多重共线性
    X_vif = df[independent_vars]
    vif_data = pd.DataFrame()
    vif_data["变量"] = X_vif.columns
    vif_data["VIF值"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print("\n多重共线性检验 (VIF):")
    print(vif_data.sort_values("VIF值", ascending=False))
    
    # 构建回归模型
    X = sm.add_constant(df[independent_vars])  # 添加常数项
    y = df["Y染色体浓度"]
    
    # 初步模型
    model = sm.OLS(y, X).fit()
    print("\n===== 初步回归模型结果 =====")
    print(model.summary())
    
    # 逐步回归优化模型（基于p值）
    significant_vars = [var for var in independent_vars if model.pvalues[var] < 0.05]
    print(f"\n显著变量 (p < 0.05): {significant_vars}")
    
    # 构建优化后的模型
    if significant_vars:
        X_opt = sm.add_constant(df[significant_vars])
        model_opt = sm.OLS(y, X_opt).fit()
        print("\n===== 优化后的回归模型结果 =====")
        print(model_opt.summary())
        
        # 残差分析
        residuals_analysis(model_opt, y)
        return model_opt, significant_vars
    else:
        print("没有显著的自变量，无法构建有效模型")
        return model, independent_vars

# 8. 残差分析
def residuals_analysis(model, y):
    """回归模型残差分析"""
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    plt.figure(figsize=(12, 8))
    
    # 残差正态性检验
    plt.subplot(221)
    stats.probplot(residuals, plot=plt)
    plt.title("残差Q-Q图")
    plt.xlabel("理论分位数")
    plt.ylabel("样本分位数")
    
    # 残差直方图
    plt.subplot(222)
    sns.histplot(residuals, kde=True)
    plt.title("残差分布直方图")
    plt.xlabel("残差")
    
    # 残差与拟合值关系图
    plt.subplot(223)
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("拟合值")
    plt.ylabel("残差")
    plt.title("残差 vs 拟合值")
    
    # 残差自相关图
    plt.subplot(224)
    sm.graphics.tsa.plot_acf(residuals, lags=10, ax=plt.gca())
    plt.title("残差自相关图")
    plt.xlabel("滞后")
    
    plt.tight_layout()
    plt.savefig("residual_analysis.png", dpi=300)
    plt.close()
    print("残差分析图已保存为 residual_analysis.png")
    
    # 残差正态性检验
    shapiro_test = stats.shapiro(residuals)
    print(f"\n残差正态性检验 (Shapiro-Wilk): statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue > 0.05:
        print("残差符合正态分布 (p > 0.05)")
    else:
        print("残差不符合正态分布 (p <= 0.05)")

# 9. 模型解释与可视化（增加GC含量相关图表）
def visualize_regression_results(model, significant_vars, df):
    """可视化回归结果，包含GC含量（如果显著）"""
    if model is None:
        return
    
    # 回归系数可视化
    coefficients = model.params.drop("const") if "const" in model.params else model.params
    plt.figure(figsize=(10, 6))
    coefficients.sort_values().plot(kind="barh")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("回归系数及其方向")
    plt.xlabel("系数值")
    plt.tight_layout()
    plt.savefig("regression_coefficients.png", dpi=300)
    plt.close()
    print("回归系数图已保存为 regression_coefficients.png")
    
    # GC含量的效应图（如果显著）
    if "GC含量" in significant_vars and "GC含量" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(x="GC含量", y="Y染色体浓度", data=df,
                   scatter_kws={"alpha":0.3}, line_kws={"color":"green"})
        plt.title(f"Y染色体浓度与GC含量的关系 (系数: {model.params['GC含量']:.4f})")
        plt.tight_layout()
        plt.savefig("gc_content_effect.png", dpi=300)
        plt.close()
        print("GC含量效应图已保存为 gc_content_effect.png")
    
    # 妊娠类型的效应图（如果显著）
    if "妊娠类型" in significant_vars and "妊娠类型" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="妊娠类型", y="Y染色体浓度", data=df)
        plt.xticks([0, 1, 2], ["自然受孕", "IUI", "IVF"])
        plt.title(f"不同妊娠类型的Y染色体浓度 (系数: {model.params['妊娠类型']:.4f})")
        plt.tight_layout()
        plt.savefig("pregnancy_type_effect.png", dpi=300)
        plt.close()
        print("妊娠类型效应图已保存为 pregnancy_type_effect.png")
    
    # 其他显著变量的边际效应图
    if "检测孕周" in significant_vars and "检测孕周" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(x="检测孕周", y="Y染色体浓度", data=df, 
                   scatter_kws={"alpha":0.3}, line_kws={"color":"red"})
        plt.title(f"Y染色体浓度与检测孕周的关系 (系数: {model.params['检测孕周']:.4f})")
        plt.tight_layout()
        plt.savefig("gestational_age_effect.png", dpi=300)
        plt.close()
        print("孕周效应图已保存为 gestational_age_effect.png")
    
    if "孕妇BMI" in significant_vars and "孕妇BMI" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(x="孕妇BMI", y="Y染色体浓度", data=df, 
                   scatter_kws={"alpha":0.3}, line_kws={"color":"blue"})
        plt.title(f"Y染色体浓度与孕妇BMI的关系 (系数: {model.params['孕妇BMI']:.4f})")
        plt.tight_layout()
        plt.savefig("bmi_effect.png", dpi=300)
        plt.close()
        print("BMI效应图已保存为 bmi_effect.png")

# 主函数
def main(file_path):
    # 加载数据
    df = load_data(file_path)
    if df is None:
        return
    
    # 数据预处理（不筛选GC含量）
    df_processed = preprocess_data(df)
    
    # 探索性数据分析
    exploratory_analysis(df_processed)
    
    # 构建回归模型
    model, significant_vars = build_regression_model(df_processed)
    
    # 可视化回归结果
    visualize_regression_results(model, significant_vars, df_processed)
    
    print("\n分析完成！所有结果图表已保存为PNG文件。")

if __name__ == "__main__":
    # 替换为你的Excel数据文件路径
    data_file_path = "附件.xlsx"  # 请将此处替换为实际Excel文件路径
    main(data_file_path)
    
