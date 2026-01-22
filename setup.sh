#!/bin/bash

# AI Alpha Policy - Setup and Quick Start Script

echo "=========================================="
echo "AI Alpha Policy - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p checkpoints
mkdir -p runs
mkdir -p backtest_results

# Generate sample data
echo ""
echo "Generating sample trading data..."
python generate_sample_data.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. View quick start guide:"
echo "   python QUICKSTART.py"
echo ""
echo "2. Configure your settings:"
echo "   nano config.yaml"
echo ""
echo "3. Train the model:"
echo "   python train.py --config config.yaml --data_path ./data/human_trades.csv"
echo ""
echo "4. Run backtest:"
echo "   python backtest.py --model_path ./checkpoints/best_model.pt --start_date 2023-01-01 --end_date 2024-01-01"
echo ""
echo "5. Start paper trading:"
echo "   python live_trading.py --model_path ./checkpoints/best_model.pt --mode paper"
echo ""
echo "=========================================="

