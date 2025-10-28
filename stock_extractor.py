def extract_and_standardize_symbol(user_query: str) -> str:
    query = user_query.strip().upper()
    mapping = {
        "VODAFONE": "IDEA",
        "RELIANCE": "RELIANCE",
        "HDFC": "HDFC",
        "INFY": "INFY",
        "TCS": "TCS"
    }
    for key, symbol in mapping.items():
        if key in query:
            return symbol
    return 