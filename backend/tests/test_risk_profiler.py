from ml.risk_profiler import (
    calculate_user_risk_profile,
    get_calculated_risk_score
)

# Old calling style
result = calculate_user_risk_profile(
    age=30,
    investment_horizon_years=10,
    savings_to_income_ratio=0.5,
    max_loss_amount=50000,
    portfolio_value=500000
)

print(result["continuous_risk_score_Ru"])
print(result["discrete_risk_score"])
print(result["category"])

# Old optimizer-facing function
score = get_calculated_risk_score(
    investment_horizon_years=10,
    max_loss_amount=50000
)

print(score)
assert isinstance(score, int)
assert 1 <= score <= 10

# Confirm original keys remain present
old_keys = {
    "continuous_risk_score_Ru",
    "discrete_risk_score",
    "category",
    "max_loss_tolerance_pct",
    "max_loss_amount",
    "investment_horizon_years",
    "portfolio_value",
    "age",
}

assert old_keys.issubset(result.keys())

print("Backward compatibility test passed!")