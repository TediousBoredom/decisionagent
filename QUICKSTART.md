# AlphaPolicy 快速入门指南

## 项目简介

AlphaPolicy 是一个基于扩散模型的全自动交易系统，能够从市场中已验证的高收益策略中学习，并在不同市场状态下自主执行交易决策。

## 核心理念

**将高收益策略转化为 AI 可执行的自动化系统**

通过对高收益策略在不同市场状态下的风险暴露与行为分布进行建模，构建策略约束与风险控制模块，使 AI 在无需人工规则干预的情况下，实现稳定盈利的实盘交易。

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **扩散模型**: DDPM/DDIM
- **数据处理**: Pandas, NumPy
- **技术分析**: TA-Lib, pandas-ta
- **交易接口**: CCXT, Alpaca API
- **可视化**: Matplotlib, Plotly
- **实验跟踪**: Weights & Biases

## 安装步骤

### 1. 克隆项目（如果需要）
```bash
cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/alphapolicy
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量（可选，用于实盘交易）
```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_API_SECRET="your_api_secret"
```

## 快速开始

### 步骤 1: 运行演示
```bash
python examples/demo.py
```

这将展示：
- 策略生成过程
- 数据处理流程
- 风险管理机制

### 步骤 2: 准备训练数据

系统会自动下载数据，或者你可以准备自己的数据：

```python
from data.dataset import DataProcessor

processor = DataProcessor()
df = processor.download_data(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01',
    source='yahoo'
)

# 处理数据并添加技术指标
df_processed = processor.process_ohlcv(df)
df_processed.to_csv('data/train_data.csv', index=False)
```

### 步骤 3: 训练模型

```bash
python examples/train.py \
    --config configs/train_config.yaml \
    --data_dir data \
    --checkpoint_dir checkpoints \
    --wandb
```

训练参数可在 `configs/train_config.yaml` 中调整。

### 步骤 4: 回测评估

```bash
python examples/trade.py \
    --model checkpoints/best_model.pt \
    --mode backtest \
    --symbol AAPL \
    --initial_capital 100000
```

### 步骤 5: 模拟交易（可选）

```bash
python examples/trade.py \
    --model checkpoints/best_model.pt \
    --mode paper \
    --symbol AAPL
```

## 项目结构

```
alphapolicy/
├── models/              # 神经网络模型
│   ├── diffusion.py    # 扩散模型核心
│   ├── encoder.py      # 市场状态编码器
│   └── policy_net.py   # 策略网络
├── data/               # 数据处理
│   └── dataset.py      # 数据集和特征工程
├── risk/               # 风险管理
│   └── constraints.py  # 风险约束和控制
├── strategy/           # 策略生成
│   └── generator.py    # 策略生成器
├── execution/          # 交易执行
│   └── engine.py       # 执行引擎
├── training/           # 训练流程
│   └── trainer.py      # 训练器
├── examples/           # 示例脚本
│   ├── demo.py        # 快速演示
│   ├── train.py       # 训练脚本
│   └── trade.py       # 交易脚本
└── configs/           # 配置文件
    ├── train_config.yaml
    └── trade_config.yaml
```

## 核心概念

### 1. 扩散模型用于策略生成

扩散模型通过学习高收益策略的分布，能够生成适应不同市场状态的交易策略：

```python
# 生成策略
output = model(
    price, indicators, orderbook, regime,
    current_position, portfolio_value, current_drawdown,
    num_samples=3,
    return_best=True
)

actions = output['actions']  # 交易动作序列
```

### 2. 交易动作定义

每个交易动作包含 4 个维度：
- **position**: 目标仓位 [-1, 1]（-1=满仓做空，1=满仓做多）
- **urgency**: 执行紧急度 [0, 1]（0=被动，1=激进）
- **stop_loss**: 止损距离 [0, 1]
- **take_profit**: 止盈距离 [0, 1]

### 3. 风险控制

多层次风险管理：

```python
# 定义风险约束
constraints = RiskConstraints(
    max_position_size=1.0,      # 最大仓位
    max_leverage=2.0,           # 最大杠杆
    max_drawdown=0.15,          # 最大回撤 15%
    max_daily_loss=0.03,        # 最大日损失 3%
    min_sharpe_ratio=1.5        # 最小夏普比率
)
```

### 4. 策略蒸馏

从历史高收益策略中学习：

```python
# 训练步骤
losses = model.train_step(
    price, indicators, orderbook, regime,
    expert_actions,  # 高收益策略的动作
    returns,         # 实际收益
    volatility       # 实际波动率
)
```

## 配置说明

### 训练配置 (`configs/train_config.yaml`)

```yaml
# 数据设置
symbols: ["AAPL", "MSFT", "GOOGL"]
start_date: "2018-01-01"
end_date: "2024-01-01"

# 模型参数
hidden_dim: 256
num_timesteps: 1000

# 训练参数
batch_size: 32
num_epochs: 100
learning_rate: 0.0001

# 风险约束
max_drawdown: 0.2
max_daily_loss: 0.05
```

### 交易配置 (`configs/trade_config.yaml`)

```yaml
# 风险约束
max_position_size: 1.0
max_drawdown: 0.15

# 执行设置
commission_rate: 0.001
slippage: 0.001

# 策略设置
num_samples: 3
use_ddim: true
```

## 性能监控

系统自动跟踪以下指标：

- **收益指标**: 总收益率、年化收益率
- **风险指标**: 最大回撤、波动率、VaR
- **风险调整收益**: 夏普比率、索提诺比率
- **交易指标**: 胜率、盈亏比、交易次数

## 常见问题

### Q1: 如何添加新的技术指标？

在 `models/encoder.py` 的 `FeatureExtractor` 类中添加：

```python
def _calculate_new_indicator(self, data):
    # 你的指标计算逻辑
    return indicator_values
```

### Q2: 如何调整风险参数？

修改配置文件中的风险约束，或在代码中创建 `RiskConstraints` 对象时指定。

### Q3: 支持哪些交易所？

目前支持：
- Yahoo Finance（历史数据）
- Binance（加密货币）
- Alpaca（美股）
- 可通过 CCXT 扩展到其他交易所

### Q4: 如何进行实盘交易？

⚠️ **警告**: 实盘交易有风险，请务必：
1. 先进行充分的回测
2. 在模拟环境测试
3. 从小资金开始
4. 设置严格的风险限制

## 注意事项

1. **风险提示**: 本系统仅供研究和学习使用，实盘交易存在风险
2. **数据质量**: 确保使用高质量的市场数据
3. **过拟合**: 避免在训练数据上过度优化
4. **监管合规**: 遵守当地金融监管法规
5. **持续监控**: 实盘运行时需要持续监控系统状态

## 进阶使用

### 自定义策略评估

```python
from risk.constraints import RiskMetrics

metrics = RiskMetrics()
sharpe = metrics.sharpe_ratio(returns)
max_dd = metrics.max_drawdown(returns)
```

### 集成多个模型

```python
from strategy.generator import StrategyEnsemble

ensemble = StrategyEnsemble(num_models=3)
output = ensemble(price, indicators, orderbook, regime, ...)
```

### 在线学习

```python
from training.trainer import OnlineTrainer

online_trainer = OnlineTrainer(model)
online_trainer.add_experience(...)
online_trainer.update(num_updates=10)
```

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**祝交易顺利！但请记住：过去的表现不代表未来的结果。**

