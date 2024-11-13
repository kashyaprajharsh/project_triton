from llms import llm
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
from config import setup_environment, news_client_id, FINANCIAL_MODELING_PREP_API_KEY, apha_api_key

setup_environment()




er = EventRegistry(apiKey = news_client_id, allowUseOfArchive=False)


@tool
def get_stock_price(symbol):
    """
    Fetch the current stock price for the given symbol, the current volume, the average price 50d and 200d, EPS, PE and the next earnings Announcement.
    """
    url = f"https://financialmodelingprep.com/api/v3/quote-order/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        price = data[0]['price']
        volume = data[0]['volume']
        priceAvg50 = data[0]['priceAvg50']
        priceAvg200 = data[0]['priceAvg200']
        eps = data[0]['eps']
        pe = data[0]['pe']
        earningsAnnouncement = data[0]['earningsAnnouncement']
        return {"symbol": symbol.upper(), "price": price, "volume":volume,"priceAvg50":priceAvg50, "priceAvg200":priceAvg200, "EPS":eps, "PE":pe, "earningsAnnouncement":earningsAnnouncement }
    except (IndexError, KeyError):
        return {"error": f"Could not fetch price for symbol: {symbol}"}

@tool
def get_company_financials(symbol):
    """
    Fetch basic financial information for the given company symbol such as the industry, the sector, the name of the company, and the market capitalization.
    """
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        results = data[0]
        financials = {
            "symbol": results["symbol"],
            "companyName": results["companyName"],
            "marketCap": results["mktCap"],
            "industry": results["industry"],
            "sector": results["sector"],
            "website": results["website"],
            "beta":results["beta"],
            "price":results["price"],
        }
        return financials
    except (IndexError, KeyError):
        return {"error": f"Could not fetch financials for symbol: {symbol}"}

@tool
def get_income_statement(symbol):
    """
    Fetch last income statement for the given company symbol such as revenue, gross profit, net income, EBITDA, EPS.
    """
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    try:
        results = data[0]
        financials = {
            "date": results["date"],
            "revenue": results["revenue"],
            "gross profit": results["grossProfit"],
            "net Income": results["netIncome"],
            "ebitda": results["ebitda"],
            "EPS": results["eps"],
            "EPS diluted":results["epsdiluted"]
        }
        return data, financials
    except (IndexError, KeyError):
        return {"error": f"Could not fetch financials for symbol: {symbol}"}
    

@tool
def get_balance_sheet(symbol):
    """
    Fetch the balance sheet statement for a given company symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        dict: Balance sheet data including assets, liabilities, and equity information
    """
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        
        # Get most recent statement
        latest = data[0]
        
        # Extract key metrics
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
            
            # Ratios (calculated)
            "current_ratio": round(latest["totalCurrentAssets"] / latest["totalCurrentLiabilities"], 2),
            "debt_to_equity": round(latest["totalDebt"] / latest["totalStockholdersEquity"], 2) if latest["totalStockholdersEquity"] != 0 else None
        }
        
        return data, financials
    
    except Exception as e:
        return {"error": f"Could not fetch balance sheet for {symbol}. Error: {str(e)}"}
    


@tool
def get_cash_flow(symbol):
    """
    Fetch the cash flow statement for a given company symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        dict: Cash flow data including operating, investing, and financing activities
    """
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    
    try:
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
        return {"error": f"Could not fetch cash flow statement for {symbol}. Error: {str(e)}"}
    



polygon = PolygonAPIWrapper()
ptoolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)

tools = ptoolkit.get_tools()
# print(tools)
# for tool in tools:
#     print(tool.name)
polygon_agg_tool = next(tool for tool in tools if tool.name == "polygon_aggregates")
polygon_ticker_news_tool = next(tool for tool in tools if tool.name == "polygon_ticker_news")
polygon_financials_tool = next(tool for tool in tools if tool.name == "polygon_financials")


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
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={apha_api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            return {"error": data["Error Message"]}
            
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
        return {"error": f"Failed to fetch news sentiment: {str(e)}"}
    
@tool
def get_insider_transactions(symbol: str) -> dict:
    """
    Fetch and summarize insider transactions for a given stock symbol.
    """
    try:
        url = f'https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={symbol}&apikey={apha_api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            return {"error": data["Error Message"]}
        
        # Process transactions
        transactions = []
        total_buys = 0
        total_sells = 0
        total_buy_value = 0
        total_sell_value = 0
        
        for transaction in data.get('data', []):
            try:
                # Safely convert values with error handling
                shares = float(transaction['shares'] or 0)
                price = float(transaction['share_price'] or 0)
                value = shares * price
                
                # Track buy/sell totals
                if transaction['acquisition_or_disposal'] == 'A':
                    total_buys += 1
                    total_buy_value += value
                else:
                    total_sells += 1
                    total_sell_value += value
                
                # Only add transaction if it has valid values
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
                continue  # Skip this transaction if there's any parsing error
        
        # Only return significant transactions
        significant_transactions = sorted(
            transactions,  # Already filtered for non-zero values
            key=lambda x: x['value'], 
            reverse=True
        )[:10]
        
        if not significant_transactions:
            return {
                "message": "No significant insider transactions found in the recent data.",
                "transaction_summary": {
                    'total_buys': total_buys,
                    'total_sells': total_sells,
                    'total_buy_value': round(total_buy_value, 2),
                    'total_sell_value': round(total_sell_value, 2),
                    'net_transaction_value': round(total_buy_value - total_sell_value, 2)
                }
            }
            
        return {
            'recent_transactions': significant_transactions,
            'transaction_summary': {
                'total_buys': total_buys,
                'total_sells': total_sells,
                'total_buy_value': round(total_buy_value, 2),
                'total_sell_value': round(total_sell_value, 2),
                'net_transaction_value': round(total_buy_value - total_sell_value, 2)
            }
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch insider transactions: {str(e)}",
            "message": "This might be due to API rate limiting or temporary service unavailability."
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
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={apha_api_key}'
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            return {"error": data["Error Message"]}
            
        # Process annual earnings
        annual_eps = []
        for entry in data.get('annualEarnings', [])[:5]:  # Last 5 years
            annual_eps.append({
                'year': entry['fiscalDateEnding'][:4],
                'eps': float(entry['reportedEPS'])
            })
            
        # Process quarterly earnings
        quarterly_earnings = []
        for entry in data.get('quarterlyEarnings', [])[:8]:  # Last 8 quarters
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
        recent_quarters = quarterly_earnings[:4]  # Last 4 quarters
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
        return {"error": f"Failed to fetch earnings data: {str(e)}"}


# 1. Financial Metrics Agent - focuses on core financial data
financial_metrics_tools = [
    get_stock_price,
    get_company_financials,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_earnings_history,
    polygon_financials_tool
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
    polygon_agg_tool
]

