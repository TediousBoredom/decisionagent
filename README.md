# AlphaPolicy: Diffusion-Based Autonomous Trading System

## 概述 (Overview)

AlphaPolicy 是一个基于扩散模型（Diffusion Models）的全自动交易系统，能够学习市场中已验证的高收益策略，并在不同市场状态下自主执行交易决策。

### 核心特性

- **扩散模型驱动**: 使用条件扩散模型生成交易策略，捕捉策略分布的复杂性
- **市场状态感知**: 动态识别市场状态（趋势、震荡、高波动等）并调整策略
- **风险约束**: 内置风险控制模块，包括仓位管理、止损止盈、最大回撤控制
- **策略蒸馏**: 从历史高收益策略中学习行为模式和风险特征
- **实盘执行**: 完整的交易执行引擎，支持多种交易所接口

## 系统架构

```
alphapolicy/
├── models/              # 扩散模型和神经网络架构
│   ├── diffusion.py    # 核心扩散模型
│   ├── policy_net.py   # 策略网络
│   └── encoder.py      # 市场状态编码器
├── data/               # 数据处理和特征工程
│   ├── market_data.py  # 市场数据获取
│   ├── features.py     # 特征提取
│   └── dataset.py      # 数据集构建
├── risk/               # 风险管理模块
│   ├── constraints.py  # 策略约束
│   ├── position.py     # 仓位管理
│   └── metrics.py      # 风险指标
├── strategy/           # 策略相关
│   ├── generator.py    # 策略生成器
│   ├── evaluator.py    # 策略评估
│   └── distillation.py # 策略蒸馏
├── execution/          # 交易执行
│   ├── engine.py       # 执行引擎
│   ├── broker.py       # 交易所接口
│   └── order.py        # 订单管理
├── training/           # 训练流程
│   ├── trainer.py      # 训练器
│   └── loss.py         # 损失函数
├── utils/              # 工具函数
│   ├── logger.py       # 日志
│   └── config.py       # 配置管理
└── examples/           # 示例脚本
    ├── train.py        # 训练示例
    └── trade.py        # 交易示例
```

## 技术原理

### 1. 扩散模型用于策略生成

系统使用条件扩散模型（Conditional Diffusion Model）来生成交易动作序列：

- **前向过程**: 逐步向高收益策略添加噪声
- **反向过程**: 从噪声中恢复策略，条件为当前市场状态
- **条件信号**: 市场特征、风险约束、目标收益等

### 2. 市场状态建模

通过深度学习编码器提取市场状态特征：
- 价格动量和趋势
- 波动率和流动性
- 订单簿深度
- 宏观经济指标

### 3. 风险控制

多层次风险管理：
- **策略级**: 最大回撤、夏普比率约束
- **仓位级**: 动态仓位调整、杠杆控制
- **订单级**: 滑点控制、执行成本优化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python examples/train.py --config configs/train_config.yaml
```

### 回测评估

```bash
python examples/backtest.py --model checkpoints/best_model.pt --data data/test_data.csv
```

### 实盘交易

```bash
python examples/trade.py --model checkpoints/best_model.pt --mode live
```

## 配置说明

主要配置文件在 `configs/` 目录下：

- `train_config.yaml`: 训练参数配置
- `model_config.yaml`: 模型架构配置
- `risk_config.yaml`: 风险控制参数
- `broker_config.yaml`: 交易所配置

## 性能指标

系统会跟踪以下关键指标：

- **收益指标**: 累计收益率、年化收益率、超额收益
- **风险指标**: 最大回撤、波动率、VaR、CVaR
- **风险调整收益**: 夏普比率、索提诺比率、卡尔玛比率
- **交易指标**: 胜率、盈亏比、交易频率

## 注意事项

⚠️ **风险提示**: 
- 本系统仅供研究和学习使用
- 实盘交易存在风险，请谨慎使用
- 建议先进行充分的回测和模拟交易
- 请遵守当地金融监管法规

## License

MIT License

