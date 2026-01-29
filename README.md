# Credit Card Approval Prediction with TOAD Scoring Engine  
# 基于 TOAD 的信用卡审批预测评分卡引擎

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TOAD](https://img.shields.io/badge/TOAD-0.6.0-orange)](https://toad.readthedocs.io/)

> **Production-Ready Credit Scoring Model with Built-in Stability Monitoring**  
> **一个内置稳定性监控的生产就绪型信用评分模型**

This project implements an end-to-end, **production-oriented credit scoring system** for automated credit card approval decisions. Built entirely on the **TOAD (Toolbox for Automated Data Science)** framework, it delivers a highly interpretable scorecard, comprehensive model validation, and robust **Population Stability Index (PSI) monitoring**—all critical for real-world risk management.

本项目实现了一个端到端、**面向生产的信用评分系统**，用于自动化信用卡审批决策。整个系统基于 **TOAD（自动化数据科学工具箱）** 框架构建，提供高可解释性的评分卡、全面的模型验证以及强大的**群体稳定性指数（PSI）监控**——这些都是实际风控场景中的关键要素。

---

## 📊 Dataset Description / 数据集描述

The analysis is based on the [Kaggle Home Credit Risk dataset](https://www.kaggle.com/rikdifos/credit-card-approval-prediction), which includes:

- **`application_record.csv`**: Static applicant information (e.g., income, family status, education).  
  **`application_record.csv`**: 申请人的静态信息（如收入、家庭状况、教育程度）。
- **`credit_record.csv`**: Historical monthly credit status over time.  
  **`credit_record.csv`**: 历史月度信用状态记录。

### Target Variable Construction / 标签定义
We adopt the industry-standard definition for high-risk customers:  
我们采用行业标准定义高风险客户：
- **Good Customer (`label = 1`)**: Maximum delinquency status in `['C', 'X', '0', '1', '2', '3']` (**No M3+ event**).  
  **好客户 (`label = 1`)**: 最大逾期状态为 `['C', 'X', '0', '1', '2', '3']`（**无 M3+ 事件**）。
- **Bad Customer (`label = 0`)**: Any occurrence of status `'4'` or `'5'` (**Delinquency ≥ 90 days, i.e., M3+**).  
  **坏客户 (`label = 0`)**: 出现过 `'4'` 或 `'5'` 状态（**逾期 ≥ 90 天，即 M3+**）。

This binary classification task aims to predict the likelihood of an applicant becoming a "Bad" customer.  
该二分类任务旨在预测申请人成为“坏客户”的可能性。

---

## 🔧 Methodology & Pipeline / 方法论与流程

The entire workflow is encapsulated in the notebook: **`kaggle+Credit+Card+Approval+Prediction-0129.ipynb`**.  
完整工作流已封装在 Notebook 中：**`kaggle+Credit+Card+Approval+Prediction-0129.ipynb`**。

### Core Steps / 核心步骤
1. **Data Preprocessing / 数据预处理**: Merge application and credit records; handle missing values.  
   合并申请表与信用记录；处理缺失值。
2. **Feature Selection / 特征筛选**: Using TOAD's `select` module with thresholds:  
   使用 TOAD 的 `select` 模块，设定阈值：
   - Missing rate < 60% （缺失率 < 60%）
   - Information Value (IV) > 0.02 （信息价值 IV > 0.02）
   - Correlation < 0.7 （相关性 < 0.7）
3. **Binning & WOE Transformation / 分箱与WOE转换**:  
   - Chi-square-based binning (`Combiner`)  
     基于卡方的分箱（`Combiner`）
   - WOE encoding (`WOETransformer`)  
     WOE 编码（`WOETransformer`）
4. **Modeling / 建模**: L2-regularized Logistic Regression.  
   L2 正则化的逻辑回归。
5. **Scorecard Generation / 评分卡生成**:  
   - Base Score: 600 （基准分：600）
   - PDO (Points to Double the Odds): 20 （分数翻倍点：20）
   - Base Odds: 1:30 （基准好坏比：1:30）
6. **Stability Monitoring / 稳定性监控**:  
   - **Model PSI**: Compares score/probability distributions between train and test sets.  
     **模型 PSI**：比较训练集与测试集的分数/概率分布。
   - **Feature PSI**: Monitors drift for each individual feature.  
     **特征 PSI**：监控每个特征的分布漂移。

---

## 📈 Model Performance & Key Insights / 模型性能与关键洞察

| Metric / 指标 | Value / 数值 |
| :--- | :--- |
| Accuracy / 准确率 | 99.69% |
| Precision / 精确率 | 99.69% |
| Recall / 召回率 | 100.00% |
| F1-Score / F1分数 | 99.84% |
| ROC-AUC | 1.0000 |
| KS Statistic / KS统计量 | 1.0000 |
| **Score PSI / 分数PSI** | **0.0000** |

> **⚠️ Critical Interpretation Note / 关键解读说明**:  
> The near-perfect AUC and KS are **artifacts of extreme class imbalance** (99.7% good vs. 0.3% bad). **Do not interpret these as indicative of true predictive power on rare events.**  
> 近乎完美的 AUC 和 KS 是**极端类别不平衡**（99.7% 好客户 vs. 0.3% 坏客户）导致的假象。**切勿将其视为对稀有事件具备真实预测能力的证据。**  
>   
> **Our primary validation focus is on / 我们的核心验证重点在于**:  
> - **Business Cost / 业务成本**: Estimated misclassification cost of **$198,600** on the test set.  
>   测试集上估算的误分类成本为 **$198,600**。  
> - **Stability / 稳定性**: Perfect PSI (0.0000) indicates no distributional shift between train/test, a strong sign of robustness.  
>   PSI 为 0.0000 表明训练/测试集无分布偏移，是模型稳健性的有力证明。

---

## 📁 Output Files / 输出文件

Upon successful execution of the `-0129` notebook, the following artifacts are generated:  
成功运行 `-0129` Notebook 后，将生成以下产出物：

- **`model/`**: Directory containing serialized model components for deployment.  
  **`model/`**: 存放用于部署的序列化模型组件目录。
  - `bin_combiner.pkl`: Feature binning rules.  
    `bin_combiner.pkl`: 特征分箱规则。
  - `woe_transformer.pkl`: WOE transformation mappings.  
    `woe_transformer.pkl`: WOE 转换映射。
  - `logistic_model.pkl`: Trained logistic regression model.  
    `logistic_model.pkl`: 训练好的逻辑回归模型。
- **`toad_scorecard.csv`**: Human-readable scorecard with feature bins, WOE values, coefficients, and final scores.  
  **`toad_scorecard.csv`**: 包含特征分箱、WOE 值、系数和最终分数的人类可读评分卡。
- **`toad_test_predictions.csv`**: Test set results including predicted labels, probabilities, and final scores.  
  **`toad_test_predictions.csv`**: 测试集结果，包含预测标签、概率和最终分数。
- **`monitoring_report.txt`**: Automated text report summarizing model/feature stability and actionable insights.  
  **`monitoring_report.txt`**: 自动生成的文本报告，汇总模型/特征稳定性及可操作建议。
- **`key_features_woe.png`**: Visualization of WOE for top IV features.  
  **`key_features_woe.png`**: 高 IV 特征的 WOE 可视化图。
- **`monitoring_summary.png`**: Dashboard for PSI monitoring (features, model, score distribution).  
  **`monitoring_summary.png`**: PSI 监控仪表盘（特征、模型、分数分布）。

---

## 🚀 Getting Started / 快速开始

### Prerequisites / 先决条件
- Python 3.8+

### Installation / 安装


### Usage / 使用方法
1. Download the dataset files (`application_record.csv`, `credit_record.csv`) from Kaggle and place them in the project root directory.  
   从 Kaggle 下载数据集文件（`application_record.csv`, `credit_record.csv`），并放入项目根目录。
2. Open and run the Jupyter notebook: **`kaggle+Credit+Card+Approval+Prediction-0129.ipynb`**.  
   打开并运行 Jupyter Notebook：**`kaggle+Credit+Card+Approval+Prediction-0129.ipynb`**。
3. Explore the generated output files for model insights, predictions, and **production-ready monitoring reports**.  
   查看生成的输出文件，获取模型洞察、预测结果和**可用于生产的监控报告**。

---

## 📝 Business Applications / 业务应用

This solution directly supports key business functions:  
本解决方案直接支持以下核心业务功能：
- **Automated Approval / 自动化审批**: Instantly approve low-risk applicants, reducing manual review costs.  
  即时批准低风险申请人，降低人工审核成本。
- **Risk-Based Pricing / 风险定价**: Offer different credit limits or interest rates based on the applicant's score.  
  根据申请人评分提供不同的信用额度或利率。
- **Portfolio Monitoring / 组合监控**: Use the PSI framework to continuously monitor model health in production and trigger retraining alerts.  
  利用 PSI 框架持续监控生产环境中模型健康状况，并触发重训练告警。

---

## ⚠️ Important Considerations / 重要注意事项

1. **Class Imbalance / 类别不平衡**: The severe imbalance necessitates careful metric interpretation and potentially advanced techniques (e.g., SMOTE, cost-sensitive learning) for real-world deployment.  
   严重的不平衡要求谨慎解读指标，在实际部署中可能需要采用高级技术（如 SMOTE、代价敏感学习）。
2. **Validation Strategy / 验证策略**: This project uses a simple train/test split. For production, **out-of-time (OOT) validation** is essential to simulate real-world performance.  
   本项目使用简单的训练/测试划分。在生产环境中，**跨时间验证（OOT）** 对模拟真实表现至关重要。
3. **Regulatory Compliance / 合规性**: Ensure the final model complies with fair lending regulations (e.g., ECOA, GDPR) by auditing for bias in protected attributes.  
   通过审计受保护属性中的偏见，确保最终模型符合公平借贷法规（如 ECOA、GDPR）。
4. **Deployment Readiness / 部署就绪性**: The saved `.pkl` files enable straightforward integration into a Flask/FastAPI service for real-time scoring.  
   保存的 `.pkl` 文件可轻松集成到 Flask/FastAPI 服务中，实现实时打分。

---

## 📜 License / 许可证
MIT

## 🙏 Acknowledgments / 致谢
- Dataset source: [Kaggle - Credit Card Approval Prediction](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)  
  数据集来源：[Kaggle - 信用卡审批预测](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)
- Core library: [TOAD Documentation](https://toad.readthedocs.io/)  
  核心库：[TOAD 官方文档](https://toad.readthedocs.io/)


