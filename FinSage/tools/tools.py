from FinSage.utils.llm import llm
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper
from eventregistry import *
import requests
import os
#from config import setup_environment, news_client_id, FINANCIAL_MODELING_PREP_API_KEY, apha_api_key,POLYGON_API_KEY
from FinSage.config.settings import (
    setup_environment,
    news_client_id,
    apha_api_key,
    POLYGON_API_KEY,
    FINANCIAL_MODELING_PREP_API_KEY  # Make sure this is imported
)
import yfinance as yf
from datetime import datetime

setup_environment()


er = EventRegistry(apiKey = news_client_id, allowUseOfArchive=False)


@tool
def get_stock_price(symbol):
    """
    Fetch the current stock price and key market data for the given symbol.
    """
    try:
        # Primary source: Financial Modeling Prep
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        result = data[0]
        return {
            "symbol": result["symbol"],
            "name": result["name"],
            "price": result["price"],
            "change": result["change"],
            "changesPercentage": result["changesPercentage"],
            "dayLow": result["dayLow"],
            "dayHigh": result["dayHigh"],
            "yearLow": result["yearLow"],
            "yearHigh": result["yearHigh"],
            "volume": result["volume"],
            "avgVolume": result["avgVolume"],
            "priceAvg50": result["priceAvg50"],
            "priceAvg200": result["priceAvg200"],
            "eps": result["eps"],
            "pe": result["pe"],
        }
    except Exception as e:
        try:
            # Fallback: yfinance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                "symbol": info.get("symbol", symbol),
                "name": info.get("longName", "N/A"),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "change": info.get("regularMarketChange"),
                "changesPercentage": info.get("regularMarketChangePercent"),
                "dayLow": info.get("dayLow"),
                "dayHigh": info.get("dayHigh"),
                "yearLow": info.get("fiftyTwoWeekLow"),
                "yearHigh": info.get("fiftyTwoWeekHigh"),
                "volume": info.get("volume"),
                "avgVolume": info.get("averageVolume"),
                "priceAvg50": info.get("fiftyDayAverage"),
                "priceAvg200": info.get("twoHundredDayAverage"),
                "eps": info.get("trailingEps"),
                "pe": info.get("trailingPE"),
                "source": "yfinance"
            }
        except Exception as yf_error:
            return {"error": f"Could not fetch price for symbol: {symbol}. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

@tool
def get_company_financials(symbol):
    """
    Fetch basic financial information for the given company symbol.
    """
    try:
        # Primary source: Financial Modeling Prep
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        results = data[0]
        return {
            "symbol": results["symbol"],
            "companyName": results["companyName"],
            "marketCap": results["mktCap"],
            "industry": results["industry"],
            "sector": results["sector"],
            "website": results["website"],
            "beta": results["beta"],
            "price": results["price"],
        }
    except Exception as e:
        try:
            # Fallback: yfinance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                "symbol": info.get("symbol", symbol),
                "companyName": info.get("longName"),
                "marketCap": info.get("marketCap"),
                "industry": info.get("industry"),
                "sector": info.get("sector"),
                "website": info.get("website"),
                "beta": info.get("beta"),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "source": "yfinance"
            }
        except Exception as yf_error:
            return {"error": f"Could not fetch financials for symbol: {symbol}. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

@tool
def get_income_statement(symbol):
    """
    Fetch last income statement for the given company symbol.
    """
    try:
        # Primary source: Financial Modeling Prep
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        results = data[0]
        financials = {
            "date": results["date"],
            "revenue": results["revenue"],
            "gross profit": results["grossProfit"],
            "net Income": results["netIncome"],
            "ebitda": results["ebitda"],
            "EPS": results["eps"],
            "EPS diluted": results["epsdiluted"]
        }
        return data, financials
    except Exception as e:
        try:
            # Fallback: yfinance
            stock = yf.Ticker(symbol)
            income_stmt = stock.income_stmt
            
            if income_stmt is None or income_stmt.empty:
                raise Exception("No income statement data available")
            
            latest = income_stmt.iloc[:, 0]  # Get most recent period
            
            financials = {
                "date": latest.name.strftime('%Y-%m-%d'),
                "revenue": float(latest.get("Total Revenue", 0)),
                "gross profit": float(latest.get("Gross Profit", 0)),
                "net Income": float(latest.get("Net Income", 0)),
                "ebitda": float(latest.get("EBITDA", 0)),
                "EPS": float(latest.get("Basic EPS", 0)),
                "EPS diluted": float(latest.get("Diluted EPS", 0)),
                "source": "yfinance"
            }
            
            return [{"raw": income_stmt.to_dict()}], financials
        except Exception as yf_error:
            return {"error": f"Could not fetch income statement for {symbol}. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

@tool
def get_balance_sheet(symbol):
    """
    Fetch the balance sheet statement for a given company symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        dict: Balance sheet data including assets, liabilities, and equity information
    """
    try:
        # Primary source: Financial Modeling Prep
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        latest = data[0]
        
        financials = {
            "date": latest["date"],
            "filing_date": latest["fillingDate"],
            "period": latest["period"],
            
            # Assets
            "cash_and_equivalents": latest["cashAndCashEquivalents"],
            "short_term_investments": latest["shortTermInvestments"],
            "cash_and_short_term_investments": latest["cashAndShortTermInvestments"],
            "net_receivables": latest["netReceivables"],
            "inventory": latest["inventory"],
            "total_current_assets": latest["totalCurrentAssets"],
            "total_non_current_assets": latest["totalNonCurrentAssets"],
            "total_assets": latest["totalAssets"],
            
            # Liabilities
            "accounts_payable": latest["accountPayables"],
            "short_term_debt": latest["shortTermDebt"],
            "total_current_liabilities": latest["totalCurrentLiabilities"],
            "long_term_debt": latest["longTermDebt"],
            "total_non_current_liabilities": latest["totalNonCurrentLiabilities"],
            "total_liabilities": latest["totalLiabilities"],
            
            # Equity
            "retained_earnings": latest["retainedEarnings"],
            "total_stockholders_equity": latest["totalStockholdersEquity"],
            
            # Key Metrics
            "total_debt": latest["totalDebt"],
            "net_debt": latest["netDebt"],
            
            # Ratios
            "current_ratio": round(latest["totalCurrentAssets"] / latest["totalCurrentLiabilities"], 2),
            "debt_to_equity": round(latest["totalDebt"] / latest["totalStockholdersEquity"], 2) if latest["totalStockholdersEquity"] != 0 else None
        }
        
        return data, financials
    
    except Exception as e:
        try:
            # Fallback: yfinance
            stock = yf.Ticker(symbol)
            balance_sheet = stock.balance_sheet
            
            if balance_sheet is None or balance_sheet.empty:
                raise Exception("No balance sheet data available")
            
            latest = balance_sheet.iloc[:, 0]  # Get most recent period
            
            financials = {
                "date": latest.name.strftime('%Y-%m-%d'),
                "filing_date": latest.name.strftime('%Y-%m-%d'),
                "period": "Annual",
                
                # Assets
                "cash_and_equivalents": float(latest.get("Cash And Cash Equivalents", 0)),
                "short_term_investments": float(latest.get("Short Term Investments", 0)),
                "cash_and_short_term_investments": float(latest.get("Cash And Short Term Investments", 0)),
                "net_receivables": float(latest.get("Net Receivables", 0)),
                "inventory": float(latest.get("Inventory", 0)),
                "total_current_assets": float(latest.get("Total Current Assets", 0)),
                "total_non_current_assets": float(latest.get("Total Non Current Assets", 0)),
                "total_assets": float(latest.get("Total Assets", 0)),
                
                # Liabilities
                "accounts_payable": float(latest.get("Accounts Payable", 0)),
                "short_term_debt": float(latest.get("Short Term Debt", 0)),
                "total_current_liabilities": float(latest.get("Total Current Liabilities", 0)),
                "long_term_debt": float(latest.get("Long Term Debt", 0)),
                "total_non_current_liabilities": float(latest.get("Total Non Current Liabilities", 0)),
                "total_liabilities": float(latest.get("Total Liabilities", 0)),
                
                # Equity
                "retained_earnings": float(latest.get("Retained Earnings", 0)),
                "total_stockholders_equity": float(latest.get("Total Stockholders Equity", 0)),
                
                # Key Metrics
                "total_debt": float(latest.get("Total Debt", 0)),
                "net_debt": float(latest.get("Net Debt", 0)),
                
                # Ratios
                "current_ratio": round(float(latest.get("Total Current Assets", 0)) / float(latest.get("Total Current Liabilities", 1)), 2),
                "debt_to_equity": round(float(latest.get("Total Debt", 0)) / float(latest.get("Total Stockholders Equity", 1)), 2),
                
                "source": "yfinance"
            }
            
            return [{"raw": balance_sheet.to_dict()}], financials
            
        except Exception as yf_error:
            return {"error": f"Could not fetch balance sheet for {symbol}. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

@tool
def get_cash_flow(symbol):
    """
    Fetch the cash flow statement for a given company symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        dict: Cash flow data including operating, investing, and financing activities
    """
    try:
        # Primary source: Financial Modeling Prep
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        # Get most recent statement
        latest = data[0]
        
        # Extract key metrics
        financials = {
            "date": latest["date"],
            "filing_date": latest["fillingDate"],
            "period": latest["period"],
            
            # Operating Activities
            "net_income": latest["netIncome"],
            "depreciation_amortization": latest["depreciationAndAmortization"],
            "stock_based_compensation": latest["stockBasedCompensation"],
            "working_capital_changes": latest["changeInWorkingCapital"],
            "operating_cash_flow": latest["netCashProvidedByOperatingActivities"],
            
            # Working Capital Components
            "accounts_receivable_change": latest["accountsReceivables"],
            "inventory_change": latest["inventory"],
            "accounts_payable_change": latest["accountsPayables"],
            
            # Investing Activities
            "capex": latest["investmentsInPropertyPlantAndEquipment"],
            "acquisitions": latest["acquisitionsNet"],
            "investment_purchases": latest["purchasesOfInvestments"],
            "investment_sales": latest["salesMaturitiesOfInvestments"],
            "investing_cash_flow": latest["netCashUsedForInvestingActivites"],
            
            # Financing Activities
            "debt_repayment": latest["debtRepayment"],
            "stock_repurchased": latest["commonStockRepurchased"],
            "dividends_paid": latest["dividendsPaid"],
            "financing_cash_flow": latest["netCashUsedProvidedByFinancingActivities"],
            
            # Cash Position
            "net_change_in_cash": latest["netChangeInCash"],
            "cash_end_period": latest["cashAtEndOfPeriod"],
            "cash_beginning_period": latest["cashAtBeginningOfPeriod"],
            
            # Key Metrics
            "free_cash_flow": latest["freeCashFlow"],
            "fcf_margin": round((latest["freeCashFlow"] / latest["netIncome"]) * 100, 2) if latest["netIncome"] != 0 else None
        }
        
        return data, financials
    
    except Exception as e:
        try:
            # Fallback: yfinance
            stock = yf.Ticker(symbol)
            cash_flow = stock.cashflow
            
            if cash_flow is None or cash_flow.empty:
                raise Exception("No cash flow data available")
            
            latest = cash_flow.iloc[:, 0]  # Get most recent period
            
            financials = {
                "date": latest.name.strftime('%Y-%m-%d'),
                "filing_date": latest.name.strftime('%Y-%m-%d'),
                "period": "Annual",
                
                # Operating Activities
                "net_income": float(latest.get("Net Income", 0)),
                "depreciation_amortization": float(latest.get("Depreciation & Amortization", 0)),
                "stock_based_compensation": float(latest.get("Stock Based Compensation", 0)),
                "working_capital_changes": float(latest.get("Change In Working Capital", 0)),
                "operating_cash_flow": float(latest.get("Operating Cash Flow", 0)),
                
                # Working Capital Components
                "accounts_receivable_change": float(latest.get("Change In Accounts Receivable", 0)),
                "inventory_change": float(latest.get("Change In Inventory", 0)),
                "accounts_payable_change": float(latest.get("Change In Accounts Payable", 0)),
                
                # Investing Activities
                "capex": float(latest.get("Capital Expenditure", 0)),
                "acquisitions": float(latest.get("Acquisitions Net", 0)),
                "investment_purchases": float(latest.get("Investment Purchases", 0)),
                "investment_sales": float(latest.get("Investment Sales", 0)),
                "investing_cash_flow": float(latest.get("Investing Cash Flow", 0)),
                
                # Financing Activities
                "debt_repayment": float(latest.get("Debt Repayment", 0)),
                "stock_repurchased": float(latest.get("Stock Repurchase", 0)),
                "dividends_paid": float(latest.get("Dividends Paid", 0)),
                "financing_cash_flow": float(latest.get("Financing Cash Flow", 0)),
                
                # Cash Position
                "net_change_in_cash": float(latest.get("Net Change In Cash", 0)),
                "cash_end_period": float(latest.get("Cash At End of Period", 0)),
                "cash_beginning_period": float(latest.get("Cash At Beginning of Period", 0)),
                
                # Key Metrics
                "free_cash_flow": float(latest.get("Free Cash Flow", 0)),
                "fcf_margin": round(float(latest.get("Free Cash Flow", 0)) / float(latest.get("Net Income", 1)) * 100, 2),
                
                "source": "yfinance"
            }
            
            return [{"raw": cash_flow.to_dict()}], financials
            
        except Exception as yf_error:
            return {"error": f"Could not fetch cash flow statement for {symbol}. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

polygon = PolygonAPIWrapper()
ptoolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)

tools = ptoolkit.get_tools()
# print(tools)
# for tool in tools:
#     print(tool.name)
polygon_agg_tool = next(tool for tool in tools if tool.name == "polygon_aggregates")
polygon_ticker_news_tool = next(tool for tool in tools if tool.name == "polygon_ticker_news")
# polygon_financials_tool = next(tool for tool in tools if tool.name == "polygon_financials")


@tool
def company_news(company_name: str) -> list:
    """
    Searches for a collection of news articles [most relevant, most socially shared and keyword based news] based on the given company name and industry keywords, and returns the news context as a list of articles.

    Args:
        company_name (str): The name of the company for finding relevant news articles.
        industry_keywords (list): A list of keywords related to the industry.

    Returns:
        list: A list of dictionaries containing the news articles.
    """
    news = []

    # Get most relevant news articles based on the company name
    q_pos = QueryArticlesIter(
        conceptUri=er.getConceptUri(company_name),
        categoryUri=er.getCategoryUri("investing"),
        dataType=["news"],
        lang="eng"
    )
    for art in q_pos.execQuery(er, sortBy="cosSim", maxItems=10):
        news.append(art)
    return news

@tool
def industry_news(industry_keywords: list) -> list:
    """
    Fetches the most relevant articles related to a given list of industry keywords.
    Args:
        industry_keywords (list): A list of keywords representing the industry of interest.
    Returns:
        list: A list of news articles related to the provided industry keywords.

    """
    news = []

    q_pos = QueryArticlesIter(
        conceptUri=QueryItems.OR([er.getConceptUri(i) for i in industry_keywords]),
        categoryUri=er.getCategoryUri("investing"),
        dataType=["news"],
        lang = "eng"
    )
    for art in q_pos.execQuery(er, sortBy="cosSim", maxItems=10): #sort by options - 
        news.append(art)

    return news

@tool
def get_news_sentiment(symbol: str) -> dict:
    """
    Fetch and summarize news sentiment with relevant articles for a given stock symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        dict: Summarized news sentiment data including:
            - overall_sentiment: Average sentiment across all articles
            - sentiment_distribution: Count of articles by sentiment category
            - top_tickers: Most frequently mentioned related tickers
            - key_topics: Most relevant topics from the news
            - relevant_news: List of relevant news articles with title and summary
    """
    try:
        # First attempt with Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={apha_api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise Exception(data["Error Message"])
            
        # Initialize counters
        sentiment_counts = {
            "Bearish": 0,
            "Somewhat-Bearish": 0,
            "Neutral": 0,
            "Somewhat-Bullish": 0,
            "Bullish": 0
        }
        
        ticker_mentions = {}
        topics = {}
        total_sentiment = 0
        article_count = 0
        relevant_news = []
        
        # Process each article
        for article in data.get('feed', []):
            # Check if article is relevant to the requested symbol
            is_relevant = False
            for ticker_data in article.get('ticker_sentiment', []):
                if ticker_data['ticker'] == symbol and float(ticker_data['relevance_score']) > 0.5:
                    is_relevant = True
                    break
            
            # Store relevant news
            if is_relevant:
                relevant_news.append({
                    'title': article['title'],
                    'summary': article['summary'],
                    'time_published': article['time_published'],
                    'sentiment_score': article['overall_sentiment_score'],
                    'sentiment_label': article['overall_sentiment_label']
                })
            
            # Count sentiment labels
            sentiment_counts[article['overall_sentiment_label']] += 1
            total_sentiment += article['overall_sentiment_score']
            article_count += 1
            
            # Count ticker mentions
            for ticker_data in article.get('ticker_sentiment', []):
                ticker = ticker_data['ticker']
                ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1
            
            # Count topic mentions
            for topic_data in article.get('topics', []):
                topic = topic_data['topic']
                relevance = float(topic_data['relevance_score'])
                topics[topic] = topics.get(topic, 0) + relevance

        # Prepare summary
        summary = {
            "overall_sentiment": round(total_sentiment / max(1, article_count), 3),
            "sentiment_distribution": sentiment_counts,
            "top_tickers": dict(sorted(ticker_mentions.items(), key=lambda x: x[1], reverse=True)[:5]),
            "key_topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]),
            "relevant_news": sorted(relevant_news, 
                                  key=lambda x: x['time_published'], 
                                  reverse=True)[:5]  # Get 5 most recent relevant articles
        }
            
        return summary
    except Exception as e:
        try:
            # Fallback to yfinance implementation
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                return {"error": "No news data available"}
            
            # Initialize counters for sentiment analysis
            sentiment_counts = {
                "Bearish": 0,
                "Somewhat-Bearish": 0,
                "Neutral": 0,
                "Somewhat-Bullish": 0,
                "Bullish": 0
            }
            
            relevant_news = []
            total_sentiment = 0
            article_count = 0
            ticker_mentions = {}
            
            # Process each news item
            for article in news:
                # Basic sentiment assignment based on type
                sentiment_label = "Neutral"
                sentiment_score = 0
                
                if article.get('type') == 'POSITIVE':
                    sentiment_label = "Somewhat-Bullish"
                    sentiment_score = 0.6
                elif article.get('type') == 'NEGATIVE':
                    sentiment_label = "Somewhat-Bearish"
                    sentiment_score = -0.6
                
                sentiment_counts[sentiment_label] += 1
                total_sentiment += sentiment_score
                article_count += 1
                
                relevant_news.append({
                    'title': article.get('title', ''),
                    'summary': article.get('text', ''),
                    'time_published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%dT%H:%M:%S'),
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'source': article.get('publisher', ''),
                    'url': article.get('link', '')
                })
                
                # Extract potential ticker mentions from title
                words = article.get('title', '').split()
                for word in words:
                    if word.isupper() and len(word) >= 2 and len(word) <= 5:
                        ticker_mentions[word] = ticker_mentions.get(word, 0) + 1
                
                ticker_mentions[symbol] = ticker_mentions.get(symbol, 0) + 1
            
            return {
                "overall_sentiment": round(total_sentiment / max(1, article_count), 3),
                "sentiment_distribution": sentiment_counts,
                "top_tickers": dict(sorted(ticker_mentions.items(), key=lambda x: x[1], reverse=True)[:5]),
                "article_count": article_count,
                "relevant_news": sorted(relevant_news, 
                                    key=lambda x: x['time_published'], 
                                    reverse=True)[:5],
                "source": "yfinance"
            }
                
        except Exception as yf_error:
            return {
                "error": f"Failed to fetch news data from both sources. Primary error: {str(e)}, Fallback error: {str(yf_error)}",
                "debug_info": {
                    "has_news": news is not None if 'news' in locals() else False,
                    "news_count": len(news) if 'news' in locals() and news is not None else 0
                }
            }
    
@tool
def get_insider_transactions(symbol: str) -> dict:
    """
    Fetch and summarize insider transactions for a given stock symbol.
    """
    try:
        # First attempt with Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={symbol}&apikey={apha_api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise Exception(data["Error Message"])
        
        # Process transactions
        transactions = []
        total_buys = 0
        total_sells = 0
        total_buy_value = 0
        total_sell_value = 0
        
        for transaction in data.get('data', []):
            try:
                shares = float(transaction['shares'] or 0)
                price = float(transaction['share_price'] or 0)
                value = shares * price
                
                if transaction['acquisition_or_disposal'] == 'A':
                    total_buys += 1
                    total_buy_value += value
                else:
                    total_sells += 1
                    total_sell_value += value
                
                if shares > 0 and price > 0:
                    transactions.append({
                        'date': transaction['transaction_date'],
                        'executive': transaction['executive'],
                        'title': transaction['executive_title'],
                        'type': transaction['security_type'],
                        'action': 'Buy' if transaction['acquisition_or_disposal'] == 'A' else 'Sell',
                        'shares': shares,
                        'price': price,
                        'value': value
                    })
            except (ValueError, TypeError):
                continue

        # If no transactions were found, raise an exception to trigger the fallback
        if not transactions:
            raise Exception("No transactions found in Alpha Vantage response")

        return {
            'recent_transactions': sorted(transactions, key=lambda x: x['value'], reverse=True)[:10],
            'transaction_summary': {
                'total_buys': total_buys,
                'total_sells': total_sells,
                'total_buy_value': round(total_buy_value, 2),
                'total_sell_value': round(total_sell_value, 2),
                'net_transaction_value': round(total_buy_value - total_sell_value, 2)
            },
            'source': 'alpha_vantage'
        }

    except Exception as e:
        try:
            # Fallback to yfinance
            stock = yf.Ticker(symbol)
            insider_df = stock.insider_transactions
            
            if insider_df is None or insider_df.empty:
                return {"error": "No insider transaction data available"}
            
            # Debug print to see the actual structure
            print("Column names:", insider_df.columns.tolist())
            
            transactions = []
            total_buys = 0
            total_sells = 0
            total_buy_value = 0
            total_sell_value = 0
            
            for _, row in insider_df.iterrows():
                try:
                    # Handle potential different column names
                    date = row.get('Date', row.get('date', row.get('Transaction Date', None)))
                    shares = float(row.get('Shares', row.get('shares', 0)))
                    value = abs(float(row.get('Value', row.get('value', 0))))
                    insider = row.get('Insider', row.get('insider', 'N/A'))
                    title = row.get('Title', row.get('title', 'N/A'))
                    
                    # Calculate price if possible
                    try:
                        price = value / abs(shares) if shares != 0 else 0
                    except (ZeroDivisionError, TypeError):
                        price = 0
                    
                    is_buy = shares > 0
                    if is_buy:
                        total_buys += 1
                        total_buy_value += value
                    else:
                        total_sells += 1
                        total_sell_value += value
                    
                    transactions.append({
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'executive': insider,
                        'title': title,
                        'type': 'Direct',
                        'action': 'Buy' if is_buy else 'Sell',
                        'shares': abs(shares),
                        'price': price,
                        'value': value
                    })
                except (ValueError, TypeError, AttributeError) as err:
                    print(f"Error processing row: {err}")
                    continue
            
            if not transactions:
                return {"error": "No valid transactions found in the data"}
            
            return {
                'recent_transactions': sorted(transactions, key=lambda x: x['value'], reverse=True)[:10],
                'transaction_summary': {
                    'total_buys': total_buys,
                    'total_sells': total_sells,
                    'total_buy_value': round(total_buy_value, 2),
                    'total_sell_value': round(total_sell_value, 2),
                    'net_transaction_value': round(total_buy_value - total_sell_value, 2)
                },
                'source': 'yfinance'
            }
                
        except Exception as yf_error:
            return {
                "error": f"Failed to fetch insider transactions: {str(yf_error)}",
                "debug_info": {
                    "has_insider_df": insider_df is not None if 'insider_df' in locals() else False,
                    "is_empty": insider_df.empty if 'insider_df' in locals() and insider_df is not None else True,
                    "columns": insider_df.columns.tolist() if 'insider_df' in locals() and insider_df is not None else []
                }
            }

@tool
def get_earnings_history(symbol: str) -> dict:
    """
    Fetch historical earnings data for a given stock symbol, including annual and quarterly earnings history,
    earnings surprises, and reporting dates.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'IBM', 'AAPL')
        
    Returns:
        dict: Earnings history data including:
            - Annual EPS trends
            - Quarterly earnings with surprise %
            - Recent performance metrics
            - Earnings dates and reporting times
    """
    try:
        # First attempt with Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={apha_api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise Exception(data["Error Message"])
            
        # Process annual earnings
        annual_eps = []
        for entry in data.get('annualEarnings', [])[:5]:
            annual_eps.append({
                'year': entry['fiscalDateEnding'][:4],
                'eps': float(entry['reportedEPS'])
            })
            
        # Process quarterly earnings
        quarterly_earnings = []
        for entry in data.get('quarterlyEarnings', [])[:8]:
            try:
                surprise_pct = float(entry['surprisePercentage']) if entry['surprisePercentage'] else 0
            except (ValueError, TypeError):
                surprise_pct = 0
                
            quarterly_earnings.append({
                'quarter': entry['fiscalDateEnding'],
                'reported_date': entry['reportedDate'],
                'reported_eps': float(entry['reportedEPS']),
                'estimated_eps': float(entry['estimatedEPS']),
                'surprise_pct': surprise_pct,
                'report_time': entry['reportTime']
            })
            
        # Calculate metrics
        recent_quarters = quarterly_earnings[:4]
        beats = sum(1 for q in recent_quarters if q['surprise_pct'] > 0)
        misses = sum(1 for q in recent_quarters if q['surprise_pct'] < 0)
        
        summary = {
            'annual_eps_trend': annual_eps,
            'quarterly_earnings': quarterly_earnings,
            'performance_metrics': {
                'earnings_beats_last_4q': beats,
                'earnings_misses_last_4q': misses,
                'avg_surprise_pct': sum(q['surprise_pct'] for q in recent_quarters) / len(recent_quarters),
                'next_report': quarterly_earnings[0] if quarterly_earnings else None
            }
        }
        
        return summary

    except Exception as e:
        try:
            # Fallback to yfinance
            stock = yf.Ticker(symbol)
            income_stmt = stock.income_stmt
            quarterly_income = stock.quarterly_income_stmt
            
            # Process annual earnings
            annual_eps = []
            for date, value in income_stmt.loc['Basic EPS'].items():
                annual_eps.append({
                    'year': date.strftime('%Y'),
                    'eps': float(value)
                })
                
            # Process quarterly earnings
            quarterly_earnings = []
            for date, value in quarterly_income.loc['Basic EPS'].items():
                quarterly_earnings.append({
                    'quarter': date.strftime('%Y-%m-%d'),
                    'reported_eps': float(value),
                    'estimated_eps': float(value),
                    'surprise_pct': 0,
                    'report_time': 'bmo'
                })
                
            recent_quarters = quarterly_earnings[:4]
            
            return {
                'annual_eps_trend': annual_eps,
                'quarterly_earnings': quarterly_earnings,
                'performance_metrics': {
                    'earnings_beats_last_4q': 0,
                    'earnings_misses_last_4q': 0,
                    'avg_surprise_pct': 0,
                    'next_report': quarterly_earnings[0] if quarterly_earnings else None
                },
            }
        except Exception as yf_error:
            return {"error": f"Failed to fetch earnings data from both sources. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

@tool
def get_stock_aggregates(
    symbol: str,
    multiplier: int = 1,
    timespan: str = "day",
    from_date: str = None,
    to_date: str = None,
    adjusted: bool = True,
    sort: str = "asc",
    limit: int = 5000
) -> dict:
    """
    Fetch aggregate bars (OHLCV) for a stock over a given date range with custom time windows.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
        multiplier (int): Size of the timespan multiplier (e.g., 1, 2, 5)
        timespan (str): Size of the time window ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
        from_date (str): Start date in YYYY-MM-DD format
        to_date (str): End date in YYYY-MM-DD format
        adjusted (bool): Whether to adjust for splits
        sort (str): Sort direction ('asc' or 'desc')
        limit (int): Number of results (max 50000)
    
    Returns:
        dict: Aggregated stock data including OHLCV values
    """
    try:
        # First attempt with Polygon
        base_url = "https://api.polygon.io/v2/aggs/ticker"
        url = f"{base_url}/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
            "apiKey": POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") != "OK":
            raise Exception(data.get("error", "Failed to fetch aggregate data"))
            
        results = []
        for bar in data.get("results", []):
            results.append({
                "timestamp": bar["t"],
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
                "vwap": bar.get("vw"),
                "transactions": bar.get("n")
            })
            
        return {
            "ticker": symbol,
            "adjusted": adjusted,
            "results_count": len(results),
            "aggregates": results,
        }
        
    except Exception as e:
        try:
            # Convert timespan format for yfinance
            timespan_mapping = {
                "minute": "1m",
                "hour": "1h",
                "day": "1d",
                "week": "1wk",
                "month": "1mo"
            }
            yf_timespan = timespan_mapping.get(timespan, "1d")
            
            # Fallback to yfinance
            stock = yf.Ticker(symbol)
            df = stock.history(
                start=from_date,
                end=to_date,
                interval=yf_timespan,
                actions=False,
                auto_adjust=adjusted
            )
            
            results = []
            for index, row in df.iterrows():
                results.append({
                    "timestamp": int(index.timestamp() * 1000),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                })
                
            return {
                "ticker": symbol,
                "adjusted": adjusted,
                "results_count": len(results),
                "aggregates": results,
                
            }
            
        except Exception as yf_error:
            return {"error": f"Failed to fetch aggregate data from both sources. Primary error: {str(e)}, Fallback error: {str(yf_error)}"}

# 1. Financial Metrics Agent - focuses on core financial data
financial_metrics_tools = [
    get_stock_price,
    get_company_financials,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_earnings_history,
]

# 2. News & Sentiment Agent - focuses on news analysis
news_sentiment_tools = [
    company_news,
    industry_news,
    get_news_sentiment,
    polygon_ticker_news_tool
]

# 3. Market Intelligence Agent - focuses on market data and insider activity
market_intelligence_tools = [
    get_insider_transactions,
    get_stock_aggregates
]

