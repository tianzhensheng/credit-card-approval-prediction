# Credit Card Approval Prediction with TOAD Scoring Engine  
# 基于 TOAD 的信用卡审批预测评分卡引擎

> **A production-oriented credit scoring system with built-in stability monitoring**  
> **一个内置稳定性监控的生产就绪型信用评分模型**

This project implements an end-to-end, **production-ready credit scoring system** for automated credit card approval decisions. Built on the **TOAD (Toolbox for Automated Data Science)** and **ScorecardPy** frameworks, it delivers a highly interpretable scorecard, comprehensive model validation, and robust **Population Stability Index (PSI) monitoring**—critical components for real-world risk management.

本项目实现了一个端到端、**面向生产的信用评分系统**，用于自动化信用卡审批决策。系统基于 **TOAD（自动化数据科学工具箱）** 与 **ScorecardPy** 构建，提供高可解释性的评分卡、全面的模型验证以及强大的**群体稳定性指数（PSI）监控**——这些都是实际风控场景中的核心要素。

---
## 📊 Dataset Description / 数据集描述

The analysis uses the [Kaggle Home Credit Risk dataset](https://www.kaggle.com/rikdifos/credit-card-approval-prediction):  
- **`application_record.csv`**: Static applicant information (e.g., income, family status, education).  
- **`credit_record.csv`**: Historical monthly credit status records.

分析基于 [Kaggle Home Credit Risk 数据集](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)：  
- **`application_record.csv`**: 申请人的静态信息（如收入、家庭状况、教育程度）。  
- **`credit_record.csv`**: 历史月度信用状态记录。

### Target Variable Construction / 标签定义

We define high-risk customers using an industry-standard approach:  
- **Good Customer (`label = 1`)**: Maximum delinquency status in `['C', 'X', '0', '1', '2', '3']` (**No M3+ event**).  
- **Bad Customer (`label = 0`)**: Any occurrence of status `'4'` or `'5'` (**Delinquency ≥ 90 days, i.e., M3+**).

采用行业标准定义风险标签：  
- **好客户 (`label = 1`)**: 最大逾期状态为 `['C', 'X', '0', '1', '2', '3']`（**无 M3+ 事件**）。  
- **坏客户 (`label = 0`)**: 出现过 `'4'` 或 `'5'` 状态（**逾期 ≥ 90 天，即 M3+**）。

---
## 🔧 Methodology & Pipeline / 方法论与流程

The complete workflow is in the notebook: **`kaggle+Credit+Card+Approval+Prediction-0130.ipynb`**.  
完整工作流详见 Notebook：**`kaggle+Credit+Card+Approval+Prediction-0130.ipynb`**。

### Core Steps / 核心步骤

1. **Data Preprocessing**  
   Merge application and credit records; handle missing values.  
   **数据预处理**：合并申请表与信用记录；处理缺失值。

2. **Feature Selection**  
   Using TOAD’s `select` module with thresholds:  
   - Missing rate < 60%  
   - Information Value (IV) > 0.02  
   - Correlation < 0.7  
   **特征筛选**：基于缺失率、IV 和相关性进行过滤。

3. **Binning & WOE Transformation**  
   - Chi-square-based optimal binning (`Combiner`)  
   - WOE encoding (`WOETransformer`)  
   **分箱与WOE转换**：采用卡方最优分箱与WOE编码。

4. **Modeling**  
   L2-regularized Logistic Regression.  
   **建模**：L2 正则化逻辑回归。

5. **Scorecard Generation**  
   - Base Score: 600  
   - PDO (Points to Double the Odds): 20  
   - Base Odds: 1:30  
   **评分卡生成**：设定基准分、PDO 与基准好坏比。

6. **Stability Monitoring**  
   - **Model PSI**: Compares score/probability distributions between train and test sets.  
   - **Feature PSI**: Monitors drift for each individual feature.  
   **稳定性监控**：计算模型 PSI 与各特征 PSI，评估分布稳定性。

---
## 📈 Model Performance & Key Insights / 模型性能与关键洞察

| Metric / 指标          | Value / 数值 |
| :--------------------- | :----------- |
| Accuracy / 准确率       | 99.38%       |
| Precision / 精确率     | 99.40%       |
| Recall / 召回率        | 99.98%       |
| F1-Score / F1分数      | 99.69%       |
| **ROC-AUC**            | **0.6226**   |
| **KS Statistic**       | **0.2438**   |
| **Estimated Cost**     | **$198,600** |
| **Score PSI**          | **0.0183**   |

> **Critical Interpretation Note / 关键解读说明**:  
> The model demonstrates **excellent stability** (Score PSI = 0.0183), indicating minimal distributional shift between train and test sets—a strong sign of robustness for production deployment.  
> However, its **discriminative power is foundational** (AUC = 0.62, KS = 0.24). This is expected given the extreme class imbalance (~99.5% good vs. ~0.5% bad) and the inherent difficulty of predicting rare default events from static application data alone.  
> **Our primary validation focus is on business impact and operational robustness**, not just statistical metrics.
>
> **模型展现出卓越的稳定性**（分数 PSI = 0.0183），表明训练/测试集间无显著分布偏移，是模型稳健性的有力证明。  
> **然而，其区分能力属于基础水平**（AUC = 0.62, KS = 0.24）。这源于极端的类别不平衡（约99.5%好客户 vs. 0.5%坏客户）以及仅凭静态申请数据预测稀有违约事件的固有难度。  
> **我们的核心验证重点在于业务影响和运营稳健性**，而非单纯的统计指标。

---
## 📁 Output Files / 输出文件

Running the `-0130` notebook generates these artifacts:  
成功运行 Notebook 后，将生成以下产出物：

- **`model/`**: Serialized model components (`bin_combiner.pkl`, `woe_transformer.pkl`, `logistic_model.pkl`).  
  **`model/`**: 序列化的模型组件，支持后续加载与部署。
- **`toad_scorecard.csv`**: Human-readable scorecard with feature bins, WOE values, and final scores.  
  **`toad_scorecard.csv`**: 人类可读的评分规则表，可直接交付业务或工程团队。
- **`toad_test_predictions.csv`**: Test set results (labels, probabilities, scores).  
  **`toad_test_predictions.csv`**: 测试集预测结果。
- **`monitoring_report.txt`**: Automated report on model/feature stability with actionable insights.  
  **`monitoring_report.txt`**: 自动生成的稳定性监控报告，包含可操作建议。
- **`key_features_woe.png`**: Visualization of WOE for top IV features.  
  **`key_features_woe.png`**: 高 IV 特征的 WOE 可视化。
- **`monitoring_summary.png`**: Dashboard for PSI monitoring (features, model, score distribution).  
  **`monitoring_summary.png`**: PSI 监控仪表盘。

---
## 🚀 Getting Started / 快速开始

1. Download `application_record.csv` and `credit_record.csv` from Kaggle and place them in the project root.  
   从 Kaggle 下载数据集并放入项目根目录。
2. Run the Jupyter notebook: **`kaggle+Credit+Card+Approval+Prediction-0130.ipynb`**.  
   运行主 Notebook。
3. Explore the output files for model insights and production-ready reports.  
   查看输出文件，获取模型洞察与可用于生产的监控报告。

---
## ⚠️ Important Considerations / 重要注意事项

1. **Class Imbalance**  
   The severe imbalance (~99.5% good) requires careful interpretation. Real-world deployment may benefit from cost-sensitive learning or advanced sampling techniques.  
   **类别不平衡**：需谨慎解读指标，实际部署可考虑代价敏感学习等方法。

2. **Validation Strategy**  
   This project uses a random train/test split. For production, **out-of-time (OOT) validation** is essential to assess temporal performance decay.  
   **验证策略**：当前为随机划分，生产环境必须采用**跨时间验证（OOT）**。

3. **Regulatory Compliance**  
   Final models must be audited for bias against protected attributes to comply with fair lending laws (e.g., ECOA, GDPR).  
   **合规性**：需审计模型在受保护属性上的公平性。

4. **Scalability & Big Data**  
   The current implementation uses `pandas` for clarity. The core logic (binning, WOE, scoring) is **framework-agnostic** and can be readily adapted to **PySpark** or **Dask** for large-scale data processing in a production data lake.  
   **可扩展性与大数据**：当前使用 `pandas` 仅为演示。核心逻辑设计为**框架无关**，可无缝迁移至 **PySpark** 或 **Dask** 以处理海量数据。

5. **Deployment Readiness**  
   The `.pkl` model files and `toad_scorecard.csv` provide a solid foundation for integration into a real-time scoring service (e.g., via Flask/FastAPI).  
   **部署就绪性**：产出物已为集成到实时 API 服务做好准备。

---
## 📜 License / 许可证

MIT
