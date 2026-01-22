# AlphaPolicy 系统架构文档

## 系统概述

AlphaPolicy 是一个基于扩散模型（Diffusion Models）的全自动交易系统，通过学习市场中已验证的高收益策略，实现 AI 自主交易决策。

## 核心技术架构

### 1. 扩散模型层 (Diffusion Layer)

**文件**: `models/diffusion.py`

- **TradingDiffusionModel**: 核心扩散模型
  - 实现 DDPM (Denoising Diffusion Probabilistic Models)
  - 支持 DDIM 快速采样
  - 条件生成：基于市场状态生成交易动作序列
  
- **关键特性**:
  - 前向扩散过程：向高收益策略添加噪声
  - 反向去噪过程：从噪声恢复策略
  - 余弦调度器：优化训练稳定性

### 2. 市场状态编码层 (Market State Encoder)

**文件**: `models/encoder.py`

- **MarketStateEncoder**: 多模态市场数据编码器
  - 价格时序编码（Temporal Convolution）
  - 技术指标编码
  - 订单簿特征编码
  - 市场状态分类
  
- **特征提取**:
  - 自动计算技术指标（MA, RSI, MACD, Bollinger Bands, ATR）
  - 多时间尺度特征融合
  - 注意力池化机制

### 3. 策略生成层 (Strategy Generator)

**文件**: `strategy/generator.py`

- **DiffusionStrategyGenerator**: 完整策略生成系统
  - 市场状态编码
  - 扩散模型策略生成
  - 风险过滤和约束
  - 收益/波动率预测
  
- **策略集成**:
  - 多样本生成
  - 最优策略选择
  - 集成学习支持

### 4. 风险管理层 (Risk Management)

**文件**: `risk/constraints.py`

- **RiskConstraints**: 风险约束定义
  - 最大仓位限制
  - 最大回撤控制
  - 杠杆限制
  - VaR/CVaR 约束
  
- **动态风险管理**:
  - **PositionSizer**: Kelly 准则仓位管理
  - **DrawdownController**: 回撤控制
  - **VolatilityScaler**: 波动率调整
  - **RiskMetrics**: 多维度风险指标计算

### 5. 执行引擎层 (Execution Engine)

**文件**: `execution/engine.py`

- **ExecutionEngine**: 交易执行引擎
  - 订单管理（市价单、限价单、止损止盈）
  - 仓位跟踪
  - 滑点和手续费模拟
  - 绩效统计
  
- **经纪商接口**:
  - **SimulatedBroker**: 回测模拟
  - **RealBrokerInterface**: 实盘接口（CCXT, Alpaca）

### 6. 训练系统 (Training System)

**文件**: `training/trainer.py`

- **StrategyDistillationTrainer**: 策略蒸馏训练器
  - 从高收益策略学习
  - 多任务学习（扩散 + 收益预测 + 波动率预测）
  - 学习率调度
  - 检查点管理
  
- **OnlineTrainer**: 在线学习
  - 经验回放缓冲区
  - 持续学习能力

## 数据流程

```
原始市场数据 (OHLCV)
    ↓
特征工程 (Technical Indicators)
    ↓
市场状态编码 (MarketStateEncoder)
    ↓
扩散模型生成 (TradingDiffusionModel)
    ↓
风险过滤 (RiskAwareActionFilter)
    ↓
交易执行 (ExecutionEngine)
    ↓
绩效反馈 (Performance Metrics)
```

## 训练流程

1. **数据准备**
   - 下载历史市场数据
   - 计算技术指标
   - 提取高收益策略的交易动作

2. **模型训练**
   - 扩散模型学习策略分布
   - 辅助任务：收益预测、波动率预测
   - 风险约束嵌入

3. **验证评估**
   - 生成策略质量评估
   - 风险指标计算
   - 与专家策略对比

## 推理流程

1. **实时数据获取**
   - 获取最新市场数据
   - 计算技术指标

2. **策略生成**
   - 编码市场状态
   - 扩散模型生成多个策略样本
   - 选择最优策略

3. **风险检查**
   - 检查风险约束
   - 调整仓位大小
   - 设置止损止盈

4. **订单执行**
   - 生成交易订单
   - 提交到交易所
   - 跟踪执行状态

## 关键创新点

### 1. 扩散模型用于策略生成
- 相比传统强化学习，扩散模型能更好地捕捉策略分布的多模态性
- 支持条件生成，可根据不同市场状态生成适应性策略

### 2. 多层次风险控制
- 策略级：最大回撤、夏普比率约束
- 仓位级：动态仓位调整、Kelly 准则
- 订单级：滑点控制、执行成本优化

### 3. 策略蒸馏
- 从历史高收益策略中学习
- 提取策略的本质特征而非简单模仿
- 泛化到新的市场环境

### 4. 自适应风险管理
- 根据市场状态动态调整风险参数
- 市场状态分类（趋势、震荡、高波动）
- 不同状态下的差异化策略

## 使用示例

### 训练模型
```bash
python examples/train.py --config configs/train_config.yaml --wandb
```

### 回测
```bash
python examples/trade.py \
    --model checkpoints/best_model.pt \
    --mode backtest \
    --symbol AAPL \
    --initial_capital 100000
```

### 运行演示
```bash
python examples/demo.py
```

## 性能指标

系统跟踪以下关键指标：

- **收益指标**: 累计收益率、年化收益率、超额收益
- **风险指标**: 最大回撤、波动率、VaR、CVaR
- **风险调整收益**: 夏普比率、索提诺比率、卡尔玛比率
- **交易指标**: 胜率、盈亏比、交易频率

## 扩展性

系统设计支持以下扩展：

1. **多资产交易**: 支持股票、期货、加密货币
2. **多策略集成**: 可集成多个策略模型
3. **自定义指标**: 易于添加新的技术指标
4. **交易所接口**: 支持多种交易所 API

## 安全性考虑

1. **风险限制**: 硬编码的风险约束
2. **紧急停止**: 触发条件时自动停止交易
3. **模拟测试**: 先在模拟环境充分测试
4. **监控告警**: 实时监控异常情况

## 未来改进方向

1. **强化学习集成**: 结合 RL 进行在线优化
2. **多模态数据**: 整合新闻、社交媒体情绪
3. **高频交易**: 支持更高频率的交易决策
4. **分布式训练**: 支持大规模数据训练
5. **可解释性**: 增强策略决策的可解释性

