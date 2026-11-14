
# questionnaire/risk_questionnaire.py

QUESTIONNAIRE_VERSION = "1.0"

RISK_QUESTIONS = [
    {
        "question_id": 1,
        "category": "Market crash reaction",
        "question": (
            "Imagine you have invested ₹5 lakh in a diversified "
            "equity portfolio. Within two months, the market falls "
            "by 20%, and your portfolio is now worth ₹4 lakh. "
            "What would you most likely do?"
        ),
        "options": [
            {
                "option_id": "A",
                "text": "Sell all my investments immediately to prevent further losses."
            },
            {
                "option_id": "B",
                "text": "Sell a significant portion and move the money to safer investments."
            },
            {
                "option_id": "C",
                "text": "Hold my investments and wait for the market to recover."
            },
            {
                "option_id": "D",
                "text": "Continue investing, or invest more while prices are lower."
            }
        ]
    },
    {
        "question_id": 2,
        "category": "Volatility tolerance",
        "question": (
            "You have two investment options. Option 1 offers "
            "relatively stable returns of around 6–7% annually. "
            "Option 2 offers potential returns of 12–15% annually, "
            "but with significant fluctuations and occasional losses. "
            "Which would you prefer?"
        ),
        "options": [
            {
                "option_id": "A",
                "text": "Option 1. I strongly prefer stability."
            },
            {
                "option_id": "B",
                "text": "Mostly Option 1, with a small allocation to Option 2."
            },
            {
                "option_id": "C",
                "text": "A balanced combination of both."
            },
            {
                "option_id": "D",
                "text": "Option 2. I am comfortable accepting fluctuations for potentially higher returns."
            }
        ]
    },
    {
        "question_id": 3,
        "category": "Investment conviction",
        "question": (
            "Your equity portfolio has underperformed the market "
            "for three consecutive years. What would you most likely do?"
        ),
        "options": [
            {
                "option_id": "A",
                "text": "Exit the equity market and move to safer investments."
            },
            {
                "option_id": "B",
                "text": "Sell most of my equity holdings."
            },
            {
                "option_id": "C",
                "text": "Review the portfolio and rebalance if necessary."
            },
            {
                "option_id": "D",
                "text": "Investigate the reasons for underperformance and continue investing if the long-term fundamentals remain sound."
            }
        ]
    },
    {
        "question_id": 4,
        "category": "Uncertainty tolerance",
        "question": (
            "You invest in a portfolio that has historically generated "
            "good long-term returns. However, its value fluctuates "
            "significantly, and you cannot predict what it will be "
            "worth one year from now. How would you feel?"
        ),
        "options": [
            {
                "option_id": "A",
                "text": "Very uncomfortable. I prefer predictable returns."
            },
            {
                "option_id": "B",
                "text": "Somewhat uncomfortable. I would prefer lower fluctuations."
            },
            {
                "option_id": "C",
                "text": "Comfortable, provided the investment aligns with my long-term goals."
            },
            {
                "option_id": "D",
                "text": "Very comfortable. I understand that uncertainty is part of investing."
            }
        ]
    },
    {
        "question_id": 5,
        "category": "Loss and liquidity needs",
        "question": (
            "Your portfolio is currently down 15%. Unexpectedly, "
            "you need money within the next six months. "
            "What would you most likely do?"
        ),
        "options": [
            {
                "option_id": "A",
                "text": "Sell the entire portfolio immediately."
            },
            {
                "option_id": "B",
                "text": "Sell most of the portfolio to ensure I have enough money."
            },
            {
                "option_id": "C",
                "text": "Sell only the amount required and retain the remaining investments."
            },
            {
                "option_id": "D",
                "text": "Use my emergency savings or other available funds for the expense, provided doing so is financially sensible."
            }
        ]
    }
]


def get_risk_questions():
    """
    Return the questionnaire without exposing internal scoring rules.
    """
    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "total_questions": len(RISK_QUESTIONS),
        "questions": RISK_QUESTIONS
    }
