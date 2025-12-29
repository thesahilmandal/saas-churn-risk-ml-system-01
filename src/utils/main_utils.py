from datetime import datetime

def format_ordinal_date(dt: datetime) -> str:
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    month = dt.strftime("%b").lower()  # dec, jan, etc.
    year = dt.year
    return f"{day}{suffix}_{month}_{year}"