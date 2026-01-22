# AI Alpha Policy - Diffusion-Based Automated Trading System

## 🎯 项目概述

这是一个基于 Diffusion 模型的全自动 AI 交易系统，能够学习并复现人类高收益交易策略，实现从策略发现到实盘执行的全流程自动化。


### 

本研究旨在将市场中已被验证具有高收益特性的人类交易策略直接转化为可由 AI 自主执行的全自动交易系统。

通过策略转化模块，AI 能够完整、稳定地复现人类高收益策略的操作流程，在无需人工规则干预的情况下完成策略发现、风险控制与实盘执行的全流程自动化。该系统确保 AI 在真实市场中能够保持正的风险调整后收益，实现对高收益策略的精确、可复现执行。

### 核心特性

- **策略学习**: 使用 Diffusion 模型从人类交易数据中学习高收益策略模式
- **风险控制**: 动态仓位管理、止损止盈、风险预算分配
- **实盘执行**: 支持多交易所实盘交易，低延迟订单执行
- **性能监控**: 实时监控策略表现，计算夏普比率、最大回撤等指标
- **回测系统**: 历史数据回测验证策略有效性

## 🏗️ 系统架构

```
American_alpha_policy/
├── models/                 # Diffusion 策略模型
│   ├── diffusion_policy.py    # 核心 Diffusion 策略网络
│   ├── unet.py                # U-Net 架构
│   └── noise_scheduler.py     # 噪声调度器
├── data/                   # 数据处理
│   ├── market_data.py         # 市场数据采集
│   ├── strategy_dataset.py    # 策略数据集
│   └── preprocessor.py        # 数据预处理
├── trading/                # 交易执行
│   ├── executor.py            # 交易执行引擎
│   ├── order_manager.py       # 订单管理
│   └── exchange_adapter.py    # 交易所适配器
├── risk/                   # 风险管理
│   ├── position_manager.py    # 仓位管理
│   ├── risk_controller.py     # 风险控制
│   └── portfolio.py           # 投资组合管理
├── backtest/               # 回测系统
│   ├── engine.py              # 回测引擎
│   └── metrics.py             # 性能指标
├── utils/                  # 工具函数
│   ├── logger.py              # 日志系统
│   └── config.py              # 配置管理
├── train.py                # 训练脚本
├── live_trading.py         # 实盘交易脚本
└── config.yaml             # 配置文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置系统

编辑 `config.yaml` 文件，设置交易所 API 密钥和策略参数。

### 3. 训练模型

```bash
python train.py --config config.yaml --data_path ./data/human_trades.csv
```

### 4. 回测验证

```bash
python backtest.py --model_path ./checkpoints/best_model.pt --start_date 2023-01-01 --end_date 2024-01-01
```

### 5. 实盘交易

```bash
python live_trading.py --model_path ./checkpoints/best_model.pt --mode paper  # 模拟盘
python live_trading.py --model_path ./checkpoints/best_model.pt --mode live   # 实盘
```

## 📊 Diffusion 策略原理

系统使用 Diffusion 模型学习人类交易策略的分布：

1. **前向过程**: 向人类交易动作逐步添加噪声
2. **反向过程**: 从噪声中恢复交易动作，生成策略决策
3. **条件生成**: 基于市场状态条件生成最优交易动作

这种方法能够捕捉复杂的策略模式，并在新市场环境中泛化。

## 🛡️ 风险管理

- **动态仓位**: 根据市场波动率和策略置信度调整仓位
- **止损机制**: 单笔交易最大损失限制
- **风险预算**: 日/周/月风险预算控制
- **相关性管理**: 避免高度相关资产的过度集中

## 📈 性能指标

系统实时计算以下指标：
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Maximum Drawdown)
- 胜率 (Win Rate)
- 盈亏比 (Profit Factor)
- 卡尔玛比率 (Calmar Ratio)

## ⚠️ 免责声明

本系统仅供研究和教育目的。实盘交易存在风险，请谨慎使用。作者不对任何交易损失负责。

## 📝 License

MIT License

