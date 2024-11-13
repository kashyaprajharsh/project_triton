# Financial Analysis Agent 🚀

A sophisticated multi-agent system powered by LangChain that provides comprehensive financial analysis and investment decision-making support through advanced AI techniques and real-time market data integration.

## 🌟 Key Features

### Advanced Multi-Agent Architecture
- **Supervisor Agent** 🎯
  - Orchestrates analysis workflow
  - Coordinates specialized agents
  - Ensures comprehensive coverage

- **Financial Metrics Agent** 📊
  - Core financial statement analysis
  - Valuation metrics & ratios
  - Company fundamentals
  - Industry comparisons

- **News Sentiment Agent** 📰
  - Real-time news monitoring
  - Sentiment analysis
  - Social media trends
  - Market perception tracking

- **Market Intelligence Agent** 📈
  - Technical analysis
  - Insider trading patterns
  - Market dynamics
  - Trading signals

- **SQL Agent** 🗄️
  - Historical data analysis
  - Performance metrics
  - Pattern recognition
  - Risk analytics

- **Reflection Agent** 🤔
  - Quality assurance
  - Analysis validation
  - Bias detection
  - Improvement suggestions

### Data Integration
- **Real-Time Market Data**
  - [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
  - [Alpha Vantage](https://www.alphavantage.co/)
  - [Polygon.io](https://polygon.io/)

- **News & Sentiment**
  - [Event Registry](https://eventregistry.org/)
  - Social media sentiment
  - Market news aggregation

- **Historical Data**
  - SQL database integration
  - Time series analysis
  - Pattern recognition

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- API keys for data providers

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kashyaprajharsh/project_triton.git
cd financial-agent
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Create .env file
cp .env.example .env

# Add your API keys
OPENAI_API_KEY=your_key
FINANCIAL_MODELING_PREP_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key
EVENT_REGISTRY_API_KEY=your_key
```

## 💻 Usage

### Starting the Application
```bash
streamlit run app.py
```

### Example Queries

#### Fundamental Analysis
```text
"Analyze AAPL's financial health and growth prospects"
"Compare MSFT's valuation metrics to industry peers"
"Evaluate NVDA's cash flow and debt management"
```

#### Technical Analysis
```text
"Analyze TSLA's price action and trading volumes"
"Identify support/resistance levels for AMD"
"Track insider trading patterns for META"
```

#### Market Research
```text
"What's the market sentiment for AI stocks?"
"Analyze semiconductor industry trends"
"Track institutional ownership changes in EV sector"
```

## 🔧 Project Structure

```
financial_agent/
├── agents/
│   ├── supervisor.py     # Orchestration logic
│   ├── financial.py      # Financial analysis
│   ├── sentiment.py      # News processing
│   ├── market.py         # Technical analysis
│   └── reflection.py     # Quality assurance
├── tools/
│   ├── financial_tools.py    # Financial APIs
│   ├── news_tools.py         # News APIs
│   ├── market_tools.py       # Market data
│   └── database_tools.py     # SQL operations
├── prompts/
│   └── system_prompts.py     # Agent prompts
├── config/
│   └── settings.py           # Configuration
└── app.py                    # Streamlit interface
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for the agent framework
- [OpenAI](https://openai.com/) for language models
- Data providers:
  - Financial Modeling Prep
  - Alpha Vantage
  - Polygon.io
  - Event Registry

## 📚 Documentation
<!-- 
For detailed documentation, please visit our [Wiki](wiki-link).

## 🔗 Links

- [Project Homepage](your-homepage)
- [Documentation](your-docs)
- [Issue Tracker](your-issues)
- [Release Notes](your-releases) -->