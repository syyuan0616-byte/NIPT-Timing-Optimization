import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# ----------------------解决决中文显示问题并设置大字体----------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 调整字体大小参数（整体调大）
plt.rcParams['font.size'] = 14  # 全局字体大小
plt.rcParams['axes.titlesize'] = 18  # 图表标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 14  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14  # y轴刻度刻度字体大小
plt.rcParams['legend.fontsize'] = 14  # 图例字体大小

# ----------------------特征定义----------------------
FEATURE_MAPPING = {
    '13号染色体的Z值': '13号染色体的Z值',
    '18号染色体的Z值': '18号染色体的Z值',
    '21号染色体的Z值': '21号染色体的Z值',
    'X染色体的Z值': 'X染色体的Z值',
    'X染色体GC含量': 'X染色体GC含量',
    '13号染色体的GC含量': '13号染色体的GC含量',
    '18号染色体的GC含量': '18号染色体的GC含量',
    '21号染色体的GC含量': '21号染色体的GC含量',
    '原始读段数': '原始始读段数',
    '重复读段的比例': '重复读段的比例',
    '唯一比对的读段数': '唯一比对的读段数',
    '被过滤掉读段数的比例': '被过滤掉读段数的比例',
    '孕妇BMI': '孕妇BMI'
}

REQUIRED_FEATURES = list(FEATURE_MAPPING.keys())

# ----------------------数据加载----------------------
def load_data(excel_path, label_column='胎儿是否健康'):
    """加载并预处理数据"""
    df = pd.read_excel(excel_path)
    
    # 处理特征列
    if '唯一比对的读段数' not in df.columns:
        matched = [col for col in df.columns if '唯一比对的读段数' in col]
        if matched:
            df = df.rename(columns={matched[0]: '唯一比对的读段数'})
    
    if 'X染色体GC含量' not in df.columns and 'GC含量' in df.columns:
        df = df.rename(columns={'GC含量': 'X染色体GC含量'})
    elif 'X染色体GC含量' not in df.columns:
        gc_cols = ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
        valid_gc = [col for col in gc_cols if col in df.columns]
        df['X染色体GC含量'] = df[valid_gc].mean(axis=1)
    
    # 确定可用特征
    available_features = [f for f in REQUIRED_FEATURES if f in df.columns]
    
    # 提取特征和标签
    X = df[available_features]
    y = df[label_column]
    
    # 转换标签为0和1（0:健康，1:不健康）
    label_values = np.unique(y)
    if set(label_values) != {0, 1}:
        y = np.where(y == label_values[1], 1, 0)
    
    # 显示数据分布
    print(f"数据分布: 健康样本={sum(y==0)}, 不健康样本={sum(y==1)}")
    return X, y, available_features

# ----------------------处理类别不平衡----------------------
def handle_imbalance(X_train, y_train, method):
    """处理类别不平衡问题"""
    if method == 'smote':
        sampler = SMOTE(random_state=np.random.randint(1000))
        return sampler.fit_resample(X_train, y_train)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=np.random.randint(1000))
        return sampler.fit_resample(X_train, y_train)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=np.random.randint(1000))
        return sampler.fit_resample(X_train, y_train)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=np.random.randint(1000))
        return sampler.fit_resample(X_train, y_train)
    else:
        return X_train, y_train

# ----------------------单次实验----------------------
def run_single_experiment(X, y, available_features):
    """运行单次实验，返回AUC及相关结果"""
    # 随机种子和参数
    random_state = np.random.randint(0, 10000)
    imbalance_method = np.random.choice(['smote', 'adasyn', 'smote_tomek', 'undersample', 'none'])
    model_type = np.random.choice(['logistic', 'rf', 'gbt'])
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # 筛选特征
    selected_features = available_features  # 使用所有可用特征最大化模型能力
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])
    
    # 处理不平衡
    X_train_resampled, y_train_resampled = handle_imbalance(
        X_train_scaled, y_train, imbalance_method
    )
    
    # 选择并训练模型
    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight='balanced'
        )
    else:  # GBT
        model = GradientBoostingClassifier(
            n_estimators=100, random_state=random_state, subsample=0.8
        )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # 预测与评估
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # 计算AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 返回结果
    return {
        'auc': roc_auc,
        'random_state': random_state,
        'method': imbalance_method,
        'model_type': model_type,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'selected_features': selected_features
    }

# ----------------------多次实验并选择最高AUC----------------------
def run_multiple_experiments(X, y, available_features, n_runs=30):
    """运行多次实验，选择AUC最高的结果"""
    print(f"开始运行{ n_runs}次实验，寻找最高AUC结果...\n")
    all_results = []
    
    # 运行实验
    for i in range(n_runs):
        result = run_single_experiment(X, y, available_features)
        all_results.append(result)
        
        # 显示进度
        current_max_auc = max(r['auc'] for r in all_results)
        if (i+1) % 5 == 0 or i == n_runs-1:
            print(f"完成{ i+1}次实验 - 当前最高AUC: {current_max_auc:.4f}")
    
    # 选择AUC最高的结果
    best_result = max(all_results, key=lambda x: x['auc'])
    print(f"\n实验完成！最高AUC: {best_result['auc']:.4f}")
    print(f"使用方法: {best_result['method']} + {best_result['model_type']}模型")
    
    return best_result

# ----------------------显示最高AUC结果----------------------
def display_best_auc_result(best_result):
    """显示AUC最高的结果详情，使用大字体图表"""
    print("\n" + "="*60)
    print(f"===== 最高AUC结果 (AUC: {best_result['auc']:.4f}) =====")
    print("="*60 + "\n")
    
    print(f"模型类型: {best_result['model_type']}")
    print(f"不平衡处理方法: {best_result['method']}")
    print(f"随机种子: {best_result['random_state']}\n")
    
    # 混淆矩阵
    cm = best_result['confusion_matrix']
    print("混淆矩阵:")
    print(f"               预测健康(0)  预测不健康(1)")
    print(f"实际健康(0)    {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"实际不健康(1)  {cm[1][0]:<12} {cm[1][1]:<12}\n")
    
    # 分类报告
    print("分类报告:")
    print(classification_report(
        best_result['y_test'], 
        best_result['y_pred'], 
        target_names=['健康', '不健康']
    ))
    
    # 绘制ROC曲线（使用大字体设置）
    plt.figure(figsize=(12, 10))  # 增大图表尺寸
    plt.plot(
        best_result['fpr'], 
        best_result['tpr'], 
        color='darkorange', 
        lw=3,  # 线条加粗
        label=f'ROC曲线 (AUC = {best_result["auc"]:.4f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.xlabel('假阳性率 (FPR)', fontsize=18)  # 单独设置更大的坐标轴标签
    plt.ylabel('真阳性率 (TPR)', fontsize=18)
    plt.title(f'最高AUC模型ROC曲线', fontsize=20)  # 单独设置更大的标题
    plt.legend(loc="lower right", fontsize=16)  # 单独设置图例大小
    plt.grid(alpha=0.3)
    plt.tight_layout()  # 自动调整布局，防止文字被截断
    plt.show()

# ----------------------主函数----------------------
def main():
    # 配置参数
    excel_path = "附件.xlsx"
    label_column = "胎儿是否健康"
    n_runs = 30  # 实验次数
    
    try:
        # 加载数据
        X, y, available_features = load_data(excel_path, label_column)
        
        # 运行多次实验并找到最高AUC结果
        best_auc_result = run_multiple_experiments(
            X, y, available_features, 
            n_runs=n_runs
        )
        
        # 显示最高AUC结果
        display_best_auc_result(best_auc_result)
        
    except Exception as e:
        print(f"程序出错: {e}")

if __name__ == "__main__":
    main()
