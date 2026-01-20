# Intro-to-ML-Project
# 📈 Market Trend Identification from Chart Images

## 📋 Project Overview
This project implements a complete machine learning pipeline for identifying market trends (Uptrend, Downtrend, Sideways) from financial chart images. The system automatically collects data, extracts features, trains multiple classifiers, and evaluates performance.

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone (https://github.com/Rana-Tarek2006/Intro-to-ML-Project)
cd market-trend-identification

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset
```python
# Generate synthetic test dataset (quick start)
python generate_dataset.py --mode synthetic --samples 50

# Or generate from real stock data
python generate_dataset.py --mode real --symbols AAPL MSFT GOOGL --start 2020-01-01 --end 2023-12-31
```

### 3. Train Model
```python
# Run complete pipeline
python train.py --data_path ./dataset --epochs 10 --batch_size 32
```

## 📁 Project Structure
```
market-trend-identification/
│
├── data/                    # Dataset directory
│   ├── Uptrend/            # Uptrend chart images
│   ├── Downtrend/          # Downtrend chart images
│   ├── Sideways/           # Sideways chart images
│   └── metadata.csv        # Dataset labels and metadata
│
├── src/                    # Source code
│   ├── data_collector.py   # Data collection and generation
│   ├── feature_extractor.py # Feature extraction methods
│   ├── classifiers.py      # ML classifiers implementation
│   ├── evaluator.py        # Model evaluation utilities
│   └── pipeline.py         # Complete training pipeline
│
├── models/                 # Trained model saves
├── results/               # Training results and plots
├── notebooks/             # Jupyter notebooks for experimentation
├── requirements.txt       # Python dependencies
├── generate_dataset.py    # Dataset generation script
├── train.py              # Training script
└── README.md             # This file
```

## 🎯 Features

### 🔍 Multi-source Data Collection
- **Real Stock Data**: Fetch from Yahoo Finance, Alpha Vantage
- **Synthetic Generation**: Create artificial chart patterns for testing
- **Web Scraping**: Collect charts from financial websites
- **Data Augmentation**: Rotate, shift, zoom for more training samples

### 📊 Feature Extraction
- **Handcrafted Features**: Statistical, texture, edge-based features
- **Deep Features**: VGG16, ResNet50, EfficientNet embeddings
- **Technical Indicators**: RSI, MACD, Bollinger Bands visualization
- **Multi-timeframe**: Support for daily, hourly, minute charts

### 🤖 Machine Learning Models
- **Traditional ML**: Random Forest, SVM, XGBoost, KNN
- **Deep Learning**: CNN, LSTM, Hybrid architectures
- **Ensemble Methods**: Voting, Stacking, Bagging
- **Transfer Learning**: Fine-tuned pre-trained models

### 📈 Evaluation Metrics
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrix Visualization**
- **ROC-AUC Curves**
- **Model Comparison Reports**
- **Feature Importance Analysis**

## 📊 Dataset Statistics

| Trend Type | Samples | Accuracy Target | Description |
|------------|---------|-----------------|-------------|
| Uptrend | 1000+ | 100% | Bullish patterns, higher highs |
| Downtrend | 1000+ | 80% | Bearish patterns, lower lows |
| Sideways | 1000+ | 60% | Consolidation, range-bound |

*Note: Accuracy targets based on typical market pattern recognition difficulty*

## 🔧 Configuration

### Data Collection Settings
```yaml
# config.yaml
data_collection:
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
  start_date: "2018-01-01"
  end_date: "2023-12-31"
  timeframe: "1d"
  window_size: 50
  prediction_window: 10
  threshold: 0.05
```

### Model Training Settings
```yaml
training:
  test_size: 0.2
  validation_size: 0.15
  random_state: 42
  n_features: 50
  classifiers: ["random_forest", "xgboost", "svm", "cnn"]
  ensemble_method: "voting"
```

## 📝 Usage Examples

### 1. Custom Dataset Creation
```python
from src.data_collector import ChartDataCollector

# Create custom dataset
collector = ChartDataCollector(
    symbols=["BTC-USD", "ETH-USD"],  # Cryptocurrency
    start_date="2020-01-01",
    timeframe="4h",  # 4-hour charts
    window_size=100
)

dataset = collector.create_dataset()
```

### 2. Feature Engineering
```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_combined_features(
    image_paths, 
    methods=['cnn', 'texture', 'edges']
)
```

### 3. Model Training
```python
from src.pipeline import TrendIdentificationPipeline

pipeline = TrendIdentificationPipeline(
    data_dir='./data',
    feature_method='cnn',
    classifiers=['rf', 'xgb', 'svm', 'ann']
)

results = pipeline.run()
```

### 4. Model Evaluation
```python
from src.evaluator import ModelEvaluator

evaluator = ModelEvaluator(results)
report = evaluator.generate_comprehensive_report()
evaluator.visualize_results(save_dir='./results')
```
## 🎨 Visualization Examples

### 1. Trend Patterns
```
Uptrend Pattern:
        ↗
      ↗
    ↗
  ↗
↗

Downtrend Pattern:
↘
  ↘
    ↘
      ↘
        ↘

Sideways Pattern:
→→→→→
  ↗↘↗↘
→→→→→
```

### 2. Generated Chart Examples
- **Candlestick Charts**: OHLC visualization
- **Line Charts**: Simplified trend lines
- **Volume Profiles**: With trading volume
- **Indicator Overlays**: RSI, MACD, Bollinger Bands

## 🔬 Research & Development

### Ongoing Improvements
1. **Multi-timeframe Analysis**: Combining daily, hourly, minute charts
2. **Market Regime Detection**: Bull/bear/sideways market identification
3. **Anomaly Detection**: Identifying unusual chart patterns
4. **Explainable AI**: Understanding model predictions
5. **Cross-asset Generalization**: Stocks, Forex, Crypto, Commodities

### Performance Optimization
- GPU acceleration with CUDA
- Batch processing for large datasets
- Model quantization for faster inference
- Distributed training with Horovod

## 📚 References

### Academic Papers
- [Chart Pattern Recognition in Financial Time Series](https://arxiv.org/abs/2106.05162)
- [Deep Learning for Financial Chart Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S095741742030528X)
- [Technical Analysis and Machine Learning](https://jfds.pm-research.com/content/early/2020/05/27/jfds.2020.1.046)

### Libraries & Tools
- **yfinance**: Stock data collection
- **mplfinance**: Financial chart plotting
- **TA-Lib**: Technical indicators
- **TensorFlow/PyTorch**: Deep learning
- **scikit-learn**: Traditional ML

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing free financial data
- TradingView for chart pattern inspiration
- The open-source ML community for tools and libraries

