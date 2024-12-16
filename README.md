# FinSage: Financial Analysis & Trading Assistant 🚀

## Overview

A sophisticated multi-agent system leveraging LangChain's agent framework and LangGraph's workflow orchestration to provide comprehensive financial analysis and investment decision-making support. What sets FinSage apart is its unique agent evaluation framework built on LangGraph's state management, ensuring reliable and validated financial insights through continuous agent performance assessment.


<div align="center">
  <img src="agent_flow.png" alt="FinSage Architecture" width="800"/>
  <p><em>FinSage Multi-Agent Architecture Diagram</em></p>
</div>

## 🌟 Why FinSage?

- **Advanced Agent Evaluation**: Built on LangGraph's state management for continuous assessment
- **Intelligent Multi-Agent System**: Powered by LangChain's agent framework for coordinated analysis
- **Real-Time Market Integration**: OpenAI GPT-4o powered data processing and analysis
- **Advanced Sentiment Analysis**: Real time market sentiment analysis
- **Interactive Visualization**: Beautiful, real-time charts powered by Streamlit
- **Production-Ready**: Enterprise-grade architecture with LangGraph error handling

## 🎯 Key Features

### Revolutionary Agent Evaluation Framework
- **LangGraph-Powered Tool Validation** 🛠️
  - Real-time monitoring through LangGraph state management
  - Invalid tool request detection via LangChain's tool validation
  - Execution error monitoring through LangGraph callbacks
  - Comprehensive tool inventory management

- **LangChain Agent Performance Metrics** 📊
  - Tool usage statistics through LangChain callbacks Handler
  - Agent specific response quality assessment
  - Task completion validation using LangGraph state tracking
 
- **Topic Adherence & Quality** 🎯
  - Task-specific validation criteria
  - Expected output verification
  - Response structure compliance
  - Source data utilization checks
  - Investment profile alignment validation

### Agent Architecture Integration
- **LangGraph Supervisor Node** 👨‍💼
  - Advanced workflow orchestration using LangGraph's StateGraph
  - Dynamic task delegation through LangChain's AgentExecutor
  - Real-time performance monitoring via LangGraph callbacks
  - Cross-agent coordination using LangGraph state management
  - Quality assurance through OpenAI evaluators

- **LangChain-Powered Specialized Agents** 🤖
  - Financial Metrics Agent with OpenAI function calling
  - Market Intelligence with LangChain tool validation
  - Sentiment Analysis with LangChain chain composition
  - SQL Analytics with structured output parsing
  - Each agent equipped with:
    - LangChain callback metrics
    - LangGraph state tracking
    - Custom evaluation checks
    - Cross-validation tools

### Evaluation Metrics & KPIs

#### LangGraph Tool Performance
```python
{
    "tool_usage": {
        "available_tools": ["financial_data", "market_analysis", "sentiment_check"],
        "used_tools": ["financial_data", "sentiment_check"],
        "unused_tools": ["market_analysis"],
        "call_counts": {"financial_data": 2, "sentiment_check": 1},
        "errors": {
            "invalid_tools": [],
            "execution_errors": [],
            "parser_errors": []
        }
    }
}
```

### Advanced Multi-Agent Architecture
- **LangGraph Orchestrator** 🎯
  - Advanced workflow orchestration using StateGraph
  - Dynamic task distribution with AgentExecutor
  - Real-time agent coordination through state management
  - Automated quality assurance via callbacks

- **Financial Metrics Agent** 📊
  - Comprehensive fundamental analysis
  - quantitative financial data analysis to evaluate a company's financial health, valuation, and performance metrics.
  - Automated financial statement analysis

- **Market Intelligence Agent** 📈
  - Real-time technical analysis
  - Volume profile analysis
  - tracks insider trading, support/resistance levels, price patterns, volume trends, and momentum indicators
 
 
- **Sentiment Analysis Agent** 📰
  - Multi-source news aggregation
  - NLP-powered sentiment scoring
  - Market sentiment indicators
  - Real-time sentiment shifts detection

- **Historical Data Analytics Agent** 🗄️
  - High-performance SQL analytics
  
### Enterprise-Grade Data Integration
- **Market Data Providers**
  - [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/) - Real-time financial data
  - [Alpha Vantage](https://www.alphavantage.co/) - Technical indicators & forex
  - [Polygon.io](https://polygon.io/) - Market data & options flow

- **News & Sentiment Sources**
  - [Event Registry](https://eventregistry.org/) - Global news monitoring
  

- **Analytics Infrastructure**
  - High-performance SQLite database
  - Real-time data processing
  - Custom analytics pipeline
  - Automated data validation

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- LangChain 
- LangGraph 
- OpenAI API access
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

####n Buy/Sell/Hold/Short/Long
```text
Should we sell or hold Alphabet Inc?
```

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
├── .streamlit/                  # Streamlit configuration
├── agents/
│   ├── __init__.py
│   ├── finsage.py              # Main agent orchestration
│   ├── financial.py            # Financial analysis agent
│   ├── sentiment.py            # News & sentiment analysis
│   ├── market.py               # Market intelligence & technical analysis
│   └── sql.py                  # Database operations & analytics
├── tools/
│   ├── __init__.py
│   ├── tools.py                # Core API integrations & utilities
│   └── plotting_tools.py       # Data visualization utilities
├── prompts/
│   ├── __init__.py
│   └── system_prompts.py       # Agent system prompts & templates
├── models/
│   ├── __init__.py
│   ├── schemas.py              # Data models & schemas
│   └── personality.py          # Agent personality definitions
├── utils/
│   ├── __init__.py
│   ├── chains.py               # LangChain utility functions
│   ├── callback_tools.py       # Callback handlers
│   └── llm/                    # LLM-specific utilities
├── config/
│   ├── __init__.py
│   ├── settings.py             # Application settings
│   └── members.py              # Team member configurations
├── .env                        # Environment variables
├── requirements.txt            # Project dependencies
├── app.py                      # Streamlit web interface
├── agent_flow.png             # Agent architecture diagram
├── stock_db.db               # SQLite database for historical data
├── langgraph.json            # Agent workflow configuration
├── trading_agentpdf.pdf      # Documentation PDF
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```


## 🔥 Advanced Features

### LangChain Integration
- Custom LangChain agents for specialized tasks
- Advanced chain composition for complex workflows
- OpenAI function calling optimization
- Custom tool development
- Structured output parsing

### LangGraph Optimization
- Efficient state management
- Parallel agent execution
- Memory-optimized workflows
- Query performance monitoring
- Real-time state tracking

### Security Features
- OpenAI API key encryption
- LangChain rate limiting
- LangGraph state validation
- Error handling and recovery
- Comprehensive audit logging




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

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)](https://github.com/hwchase17/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.1-blue)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io/)


- Premium Data Providers:
  - Financial Modeling Prep
  - Alpha Vantage
  - Polygon.io
  - Event Registry

## 📚 Documentation & Resources
<!-- 
- [Full Documentation](your-docs-link) - Comprehensive LangChain & LangGraph integration guide
- [API Reference](your-api-docs) - Detailed OpenAI function specifications
- [Video Tutorials](your-youtube) - Step-by-step LangChain implementation guides
- [Example Notebooks](your-notebooks) - LangGraph workflow examples
- [Technical Blog](your-blog-link) - Deep dives into agent evaluation -->

---
<div align="center">
  <sub>Built with ❤️ by the Guanabara AI Team using LangChain, LangGraph & OpenAI</sub>
</div>

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=kashyaprajharsh/project_triton&type=Date)](https://star-history.com/#kashyaprajharsh/project_triton&Date)

## 🤝 Community & Support

<!-- - [Join our Discord](your-discord-link) - Discuss LangChain implementations
- [Follow us on Twitter](your-twitter-link) - Latest updates on LangGraph features
- [Read our Blog](your-blog-link) - Technical deep dives into agent evaluation -->