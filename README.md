# credit-card-approval-prediction
ML-based credit risk assessment for card approval decisions
Credit Card Approval Prediction (信用卡审批预测)

📌 Overview

This project implements a credit scoring model for credit card approval prediction using two popular Python libraries: TOAD and Scorecardpy. The model leverages application records and historical credit behavior data to assess applicant risk and predict approval likelihood.

本项目使用 TOAD 和 Scorecardpy 两个流行的 Python 库实现了一个信用评分模型，用于信用卡审批预测。该模型利用申请记录和历史信用行为数据来评估申请人风险并预测审批可能性。

📊 Dataset Description (数据集描述)

The dataset is sourced from a public Kaggle contribution by user rikdifos, available under the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

The dataset consists of two main files:

- application_record.csv: Contains applicant demographic and financial information
  - ID: Unique identifier
  - CODE_GENDER: Gender (M/F)
  - FLAG_OWN_CAR: Car ownership (Y/N)
  - FLAG_OWN_REALTY: Realty ownership (Y/N)
  - CNT_CHILDREN: Number of children
  - AMT_INCOME_TOTAL: Total income
  - NAME_INCOME_TYPE: Income type (Working, Commercial associate, Pensioner, etc.)
  - NAME_EDUCATION_TYPE: Education level
  - NAME_FAMILY_STATUS: Marital status
  - NAME_HOUSING_TYPE: Housing type
  - DAYS_BIRTH: Days since birth (negative values)
  - DAYS_EMPLOYED: Days employed (negative values, positive for pensioners)
  - OCCUPATION_TYPE: Occupation type (with missing values)
  - CNT_FAM_MEMBERS: Family members count

- credit_record.csv: Contains monthly credit history
  - ID: Unique identifier (linked to application record)
  - MONTHS_BALANCE: Months balance (0 = current month, negative = past months)
  - STATUS: Credit status
    - 0, 1, 2, 3, 4, 5: Days past due (0 = no delay, 1 = 1-29 days, etc.)
    - C: Closed account
    - X: No loan for the month

数据集包含两个主要文件：

本数据集来源于 Kaggle 用户 rikdifos 的公开贡献，采用 CC0 1.0 通用（CC0 1.0）公共领域贡献协议 发布。

- application_record.csv：包含申请人人口统计和财务信息
  - ID：唯一标识符
  - CODE_GENDER：性别（M/F）
  - FLAG_OWN_CAR：汽车拥有情况（Y/N）
  - FLAG_OWN_REALTY：房产拥有情况（Y/N）
  - CNT_CHILDREN：子女数量
  - AMT_INCOME_TOTAL：总收入
  - NAME_INCOME_TYPE：收入类型（在职、商业关联、退休金等）
  - NAME_EDUCATION_TYPE：教育水平
  - NAME_FAMILY_STATUS：婚姻状况
  - NAME_HOUSING_TYPE：住房类型
  - DAYS_BIRTH：出生天数（负值）
  - DAYS_EMPLOYED：就业天数（负值，退休人员为正值）
  - OCCUPATION_TYPE：职业类型（含缺失值）
  - CNT_FAM_MEMBERS：家庭成员数量

- credit_record.csv：包含月度信用历史
  - ID：唯一标识符（与申请记录关联）
  - MONTHS_BALANCE：月度余额（0 = 当前月份，负值 = 过去月份）
  - STATUS：信用状态
    - 0, 1, 2, 3, 4, 5：逾期天数（0 = 无延迟，1 = 1-29天等）
    - C：已关闭账户
    - X：当月无贷款

🔧 Methodology (方法论)

Target Variable Construction (目标变量构建)
- Good (Label = 1): Status in ['C', 'X', '0'] (No delinquency)
- Bad (Label = 0): Status in ['1', '2', '3', '4', '5'] (Any delinquency)

Feature Engineering (特征工程)
- Missing value imputation for OCCUPATION_TYPE
- Inner join between application and credit records
- Removal of ID column

Model Development (模型开发)
Two parallel approaches were implemented:

Approach 1: TOAD Framework
1. Feature Selection: Using IV (>0.02) and correlation (0.02)
3. WOE Transformation: Built-in WOE conversion
4. Modeling: Logistic Regression with L2 regularization
5. Scoring: Standard scorecard with base score 600, PDO=20, odds=1:30

目标变量构建
- 好客户 (标签 = 1)：状态为 ['C', 'X', '0']（无逾期）
- 坏客户 (标签 = 0)：状态为 ['1', '2', '3', '4', '5']（有任何逾期）

特征工程
- 对 OCCUPATION_TYPE 进行缺失值填充
- 申请记录与信用记录内连接
- 移除 ID 列

模型开发
实现了两种并行方法：

方法1：TOAD框架
1. 特征选择：使用IV（>0.02）和相关性（0.02）
3. WOE转换：内置WOE转换
4. 建模：L2正则化的逻辑回归
5. 评分：标准评分卡，基础分600，PDO=20，好坏比=1:30

📈 Model Performance (模型性能)
Metric   TOAD Model   Scorecardpy Model
Accuracy   99.69%   99.66%

Precision   99.69%   99.66%

Recall   100.00%   100.00%

F1-Score   99.84%   99.83%

ROC-AUC   1.0000   0.5828

KS Statistic   1.0000   0.2133

PSI   0.0000   0.0010

Note: The extremely high performance metrics in the TOAD model suggest potential data leakage or overfitting issues that should be investigated in production environments.

注意：TOAD模型中极高的性能指标表明可能存在数据泄露或过拟合问题，在生产环境中应进行深入调查。

📁 Output Files (输出文件)

TOAD Implementation
- toad_scorecard.csv: Complete scorecard with feature bins, WOE values, and scores
- toad_test_predictions.csv: Test set predictions with probabilities and scores
- toad_model_coefficients.csv: Logistic regression coefficients

Scorecardpy Implementation
- scorecardpy_scorecard.csv: Complete scorecard with feature bins and scores
- scorecardpy_feature_bins.csv: Detailed binning information
- scorecardpy_test_predictions.csv: Test set predictions with probabilities and scores
- scorecardpy_feature_iv.csv: Feature Information Value statistics
- scorecardpy_model_coefficients.csv: Model coefficients and importance
- scorecardpy_score_distribution.csv: Score distribution across different ranges

TOAD实现
- toad_scorecard.csv：完整的评分卡，包含特征分箱、WOE值和分数
- toad_test_predictions.csv：测试集预测结果，包含概率和分数
- toad_model_coefficients.csv：逻辑回归系数

Scorecardpy实现
- scorecardpy_scorecard.csv：完整的评分卡，包含特征分箱和分数
- scorecardpy_feature_bins.csv：详细的分箱信息
- scorecardpy_test_predictions.csv：测试集预测结果，包含概率和分数
- scorecardpy_feature_iv.csv：特征信息值统计
- scorecardpy_model_coefficients.csv：模型系数和重要性
- scorecardpy_score_distribution.csv：不同范围的分数分布

🚀 Getting Started (快速开始)

Prerequisites (先决条件)
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, toad, scorecardpy, scipy

Installation (安装)
pip install pandas numpy scikit-learn toad scorecardpy scipy

Usage (使用方法)
1. Place your dataset files (application_record.csv, credit_record.csv) in the project directory
2. Run the Jupyter notebook kaggle+Credit+Card+Approval+Prediction.ipynb
3. Review the generated output files for model insights and predictions

先决条件
- Python 3.8+
- 所需包：pandas, numpy, scikit-learn, toad, scorecardpy, scipy

安装
pip install pandas numpy scikit-learn toad scorecardpy scipy

使用方法
1. 将数据集文件（application_record.csv, credit_record.csv）放在项目目录中
2. 运行Jupyter笔记本 kaggle+Credit+Card+Approval+Prediction.ipynb
3. 查看生成的输出文件以获取模型洞察和预测结果

📝 Business Applications (业务应用)

- Credit Risk Assessment: Evaluate applicant creditworthiness for card approval
- Automated Decision Making: Implement rule-based approval/rejection systems
- Portfolio Management: Monitor and manage credit portfolio risk
- Regulatory Compliance: Maintain transparent and explainable credit decisions

业务应用
- 信用风险评估：评估申请人信用卡审批的信用价值
- 自动化决策：实施基于规则的批准/拒绝系统
- 投资组合管理：监控和管理信贷组合风险
- 监管合规：保持透明且可解释的信贷决策

⚠️ Important Considerations (重要注意事项)

1. Data Imbalance: The dataset has severe class imbalance (99.7% good vs 0.3% bad), which requires careful handling
2. Model Validation: Cross-validation and out-of-time validation are essential for robust model assessment
3. Regulatory Requirements: Ensure compliance with fair lending laws and avoid discriminatory features
4. Production Deployment: Additional monitoring for model drift and performance degradation is necessary

重要注意事项
1. 数据不平衡：数据集存在严重的类别不平衡（99.7%好客户 vs 0.3%坏客户），需要谨慎处理
2. 模型验证：交叉验证和跨时间验证对于稳健的模型评估至关重要
3. 监管要求：确保符合公平贷款法律，避免歧视性特征
4. 生产部署：需要额外监控模型漂移和性能下降

📄 License (许可证)

This project is licensed under the MIT License - see the LICENSE file for details.

本项目采用MIT许可证 - 详情请参阅LICENSE文件。

🙏 Acknowledgments (致谢)

- Kaggle and user rikdifos for the Credit Card Approval Prediction dataset (licensed under CC0: Public Domain)
- TOAD and Scorecardpy development teams for their excellent open-source libraries
- The credit risk modeling community for sharing knowledge and best practices

致谢
- 感谢 Kaggle 及用户 rikdifos 提供的《信用卡审批预测》数据集（采用 CC0: 公共领域许可）。
- 感谢TOAD和Scorecardpy开发团队提供的优秀开源库
- 感谢信用风险建模社区分享知识和最佳实践

## 🚀 How to Download

Visit [Releases](https://github.com/tianzhensheng/credit-card-approval-prediction/releases) to download the latest version.
