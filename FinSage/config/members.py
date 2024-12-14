def get_team_members_details() -> dict:
    """
    Returns a dictionary containing details of financial agent team members.

    Each team member is represented as a dictionary with the following keys:
    - name: The name of the team member/agent
    - description: A brief description of the agent's role and capabilities
    - tools: List of primary tools used by this agent

    Returns:
        dict: A dictionary containing details of financial agent team members
    """
    members_dict = [
        {
            "name": "FinancialMetricsAgent",
            "description": "Core financial data specialist focusing on company metrics, financial statements, and fundamental analysis.",
            "tools": [
                "get_stock_price",
                "get_company_financials",
                "get_income_statement",
                "get_balance_sheet",
                "get_cash_flow",
                
            ]
        },
        {
            "name": "NewsSentimentAgent",
            "description": "Specializes in news analysis and sentiment tracking across multiple sources including company-specific news and industry trends.",
            "tools": [
                "company_news",
                "industry_news",
                "get_news_sentiment",
                "polygon_ticker_news_tool"
            ]
        },
        {
            "name": "MarketIntelligenceAgent",
            "description": "Focuses on market data analysis and insider trading patterns to provide trading insights.",
            "tools": [
                "get_insider_transactions",
                "get_stock_aggregates"
            ]
        },
        {
            "name": "SQLAgent",
            "description": "Independent LangGraph-based agent specialized in SQL database operations and historical financial data analysis. Operates as a separate workflow with its own state management.",
            "workflow_nodes": [
                "analyze_question",
                "get_schemas",
                "generate_query",
                "validate_query",
                "execute_query",
                "format_results"
            ]
        },
        # {
        #     "name": "Reflection",
        #     "description": "Critical analysis agent that reviews and improves investment analyses by evaluating completeness, insight quality, and recommendation validity.",
        #     "capabilities": [
        #         "analytical_completeness",
        #         "insight_quality",
        #         "information_integration",
        #         "recommendation_quality"
        #     ]
        # }
    ]
    return members_dict