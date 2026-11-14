
"""
Goal-Based Portfolio Optimizer
Uses historical monthly returns for eligible Indian stocks and mutual funds.

Inputs:
    backend/data/processed/monthly_asset_returns.csv
    backend/data/processed/monthly_gsec_features.csv (context only)

Outputs:
    goal_optimizer_candidates.csv
    goal_optimizer_candidate_weights.csv
    goal_optimizer_allocation.csv
    goal_optimizer_closest_candidates.csv

Important:
    Historical simulations are estimates, not guarantees.
    G-Sec yield changes are contextual information, not bond returns.
    Taxes, transaction costs, inflation, and fund/stock overlap are not modeled.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import time
from scipy.optimize import differential_evolution


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"

RETURNS_FILE = PROCESSED_DIR / "monthly_asset_returns.csv"
GSEC_FILE = PROCESSED_DIR / "monthly_gsec_features.csv"

MIN_HISTORY_MONTHS = 84
MIN_COMMON_MONTHS = 60

NUM_CANDIDATES = 3000
NUM_SIMULATIONS = 3000
RANDOM_SEED = 42

MIN_PORTFOLIO_ASSETS = 5
MAX_PORTFOLIO_ASSETS = 12

MIN_FUND_ALLOCATION = 0.20
MAX_FUND_ALLOCATION = 0.60
MAX_FUND_WEIGHT = 0.40


def money(value):
    return f"INR {value:,.2f}"


def get_stock_weight_limit(risk_score):
    """Map risk score 1-10 to the maximum weight of an individual stock."""

    if not 1 <= risk_score <= 10:
        raise ValueError("Risk score must be between 1 and 10.")

    if risk_score <= 3:
        return 0.10
    elif risk_score <= 6:
        return 0.15
    elif risk_score <= 8:
        return 0.20
    else:
        return 0.25


def load_data():
    """Load eligible assets and retain common complete monthly observations."""

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {RETURNS_FILE}"
        )

    raw = pd.read_csv(RETURNS_FILE)

    date_col = next(
        (
            col for col in ("month", "date", "Date", "Month")
            if col in raw.columns
        ),
        None,
    )

    if date_col is None:
        raise ValueError(
            "Expected a date column named 'month' or 'date'."
        )

    raw[date_col] = pd.to_datetime(
        raw[date_col], errors="coerce"
    )

    raw = raw.dropna(subset=[date_col])
    raw = raw.sort_values(date_col)
    raw = raw.rename(columns={date_col: "month"})
    raw = raw.drop_duplicates("month")
    raw = raw.set_index("month")

    all_assets = [
        col for col in raw.columns
        if col.startswith("STOCK_") or col.startswith("FUND_")
    ]

    if not all_assets:
        raise ValueError("No STOCK_ or FUND_ columns found.")

    original_stocks = [
        col for col in all_assets if col.startswith("STOCK_")
    ]

    original_funds = [
        col for col in all_assets if col.startswith("FUND_")
    ]

    eligible_assets = [
        col for col in all_assets
        if raw[col].notna().sum() >= MIN_HISTORY_MONTHS
    ]

    excluded_assets = sorted(
        set(all_assets) - set(eligible_assets)
    )

    if len(eligible_assets) < MIN_PORTFOLIO_ASSETS:
        raise ValueError(
            "Too few assets have sufficient historical data."
        )

    # Do not replace missing returns with zero.
    returns = raw[eligible_assets].dropna(axis=0, how="any")

    if len(returns) < MIN_COMMON_MONTHS:
        raise ValueError(
            f"Only {len(returns)} common complete months available."
        )

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite returns found in the dataset.")

    if (values < -1.0).any() or (values > 1.0).any():
        raise ValueError(
            "Returns outside -100% to +100% found. "
            "Check whether returns are stored as decimals."
        )

    stock_names = [
        col for col in original_stocks
        if col in returns.columns
    ]

    fund_names = [
        col for col in original_funds
        if col in returns.columns
    ]

    if not stock_names or not fund_names:
        raise ValueError(
            "The eligible data must include stocks and mutual funds."
        )

    print("\nDATA SUMMARY")
    print(f"Assets in original dataset: {len(all_assets)}")
    print(f"Eligible assets: {len(returns.columns)}")
    print(f"Eligible stocks: {len(stock_names)}")
    print(f"Eligible mutual funds: {len(fund_names)}")
    print(f"Common complete months: {len(returns)}")
    print(
        f"Date range: {returns.index.min().date()} "
        f"to {returns.index.max().date()}"
    )
    print(f"Excluded assets: {excluded_assets}")

    return returns, stock_names, fund_names


def get_gsec_context():
    """Print G-Sec context only; do not treat yield changes as returns."""

    if not GSEC_FILE.exists():
        print("\nG-Sec context file not found.")
        return

    try:
        gsec = pd.read_csv(GSEC_FILE)

        numeric_cols = gsec.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_cols:
            print("\nNo numeric G-Sec features found.")
            return

        change_cols = [
            col for col in numeric_cols
            if "change" in col.lower()
            or "delta" in col.lower()
        ]

        col = change_cols[0] if change_cols else numeric_cols[0]

        series = pd.to_numeric(
            gsec[col], errors="coerce"
        ).dropna()

        if series.empty:
            print("\nNo usable G-Sec observations found.")
            return

        print("\nG-Sec context (not investment returns)")
        print(f"Column used: {col}")
        print(
            "Yield-change standard deviation: "
            f"{series.std(ddof=1):.4f} (CSV units)"
        )
        print(
            "Latest yield-change observation: "
            f"{series.iloc[-1]:.4f} (CSV units)"
        )

    except Exception as exc:
        print(f"\nCould not read G-Sec context: {exc}")


def capped_random_weights(count, total, cap, rng):
    """Generate weights that sum to total without exceeding the cap."""

    if count <= 0:
        raise ValueError("At least one asset is required.")

    if count * cap + 1e-12 < total:
        raise ValueError("Not enough assets to satisfy the weight cap.")

    weights = rng.dirichlet(
        np.ones(count) * 1.5
    ) * total

    for _ in range(100):
        over = weights > cap

        if not over.any():
            break

        excess = float((weights[over] - cap).sum())
        weights[over] = cap

        under = ~over

        if not under.any():
            break

        under_sum = float(weights[under].sum())

        if under_sum > 0:
            weights[under] += (
                excess * weights[under] / under_sum
            )
        else:
            weights[under] += excess / under.sum()

    # Correct small numerical differences.
    if (weights > cap + 1e-8).any():
        raise ValueError("Unable to satisfy weight cap.")

    weights *= total / weights.sum()

    if (weights > cap + 1e-8).any():
        weights = np.minimum(weights, cap)
        remaining = total - weights.sum()

        for _ in range(100):
            if remaining <= 1e-10:
                break

            room = cap - weights
            available = room > 1e-10

            if not available.any():
                break

            available_room = room[available].sum()
            addition = min(remaining, float(available_room))

            weights[available] += (
                addition * room[available] / available_room
            )

            remaining = total - weights.sum()

    if abs(weights.sum() - total) > 1e-6:
        raise ValueError("Weights do not sum to the required total.")

    if (weights > cap + 1e-6).any():
        raise ValueError("Weight cap validation failed.")

    return weights


def create_candidate_weights(
    stock_names,
    fund_names,
    rng,
    max_stock_weight,
):
    """
    Create a candidate with stocks and funds.
    Mutual funds receive between 20% and 60% of the portfolio.
    """

    for _ in range(200):
        total_assets = int(
            rng.integers(
                MIN_PORTFOLIO_ASSETS,
                MAX_PORTFOLIO_ASSETS + 1,
            )
        )

        max_funds = min(len(fund_names), total_assets - 1)

        if max_funds < 1:
            continue

        fund_count = int(
            rng.integers(1, max_funds + 1)
        )

        stock_count = total_assets - fund_count

        fund_share_low = max(
            MIN_FUND_ALLOCATION,
            1.0 - stock_count * max_stock_weight,
        )

        fund_share_high = min(
            MAX_FUND_ALLOCATION,
            fund_count * MAX_FUND_WEIGHT,
        )

        if fund_share_low > fund_share_high + 1e-10:
            continue

        fund_share = float(
            rng.uniform(fund_share_low, fund_share_high)
        )

        chosen_stocks = list(
            rng.choice(
                stock_names,
                size=stock_count,
                replace=False,
            )
        )

        chosen_funds = list(
            rng.choice(
                fund_names,
                size=fund_count,
                replace=False,
            )
        )

        try:
            stock_weights = capped_random_weights(
                stock_count,
                1.0 - fund_share,
                max_stock_weight,
                rng,
            )

            fund_weights = capped_random_weights(
                fund_count,
                fund_share,
                MAX_FUND_WEIGHT,
                rng,
            )

        except ValueError:
            continue

        allocation = dict(
            zip(chosen_stocks, stock_weights)
        )

        allocation.update(
            dict(zip(chosen_funds, fund_weights))
        )

        fund_total = sum(
            weight for name, weight in allocation.items()
            if name.startswith("FUND_")
        )

        if (
            len(allocation) >= MIN_PORTFOLIO_ASSETS
            and abs(sum(allocation.values()) - 1.0) < 1e-6
            and fund_total >= MIN_FUND_ALLOCATION - 1e-6
        ):
            return allocation

    raise RuntimeError(
        "Could not generate a valid mixed stock/fund portfolio. "
        "Check the available assets and allocation caps."
    )


def simulate_portfolio(
    asset_returns,
    weights,
    sampled_month_indices,
    current_savings,
    monthly_contribution,
):
    """
    Simulate wealth using historical monthly returns sampled with replacement.
    Monthly contributions are added after each month's return.
    """

    portfolio_returns = asset_returns @ weights

    simulated_returns = portfolio_returns[
        sampled_month_indices
    ]

    wealth = np.full(
        simulated_returns.shape[0],
        current_savings,
        dtype=float,
    )

    for month_index in range(simulated_returns.shape[1]):
        wealth *= 1.0 + simulated_returns[:, month_index]
        wealth += monthly_contribution

    return wealth



def project_portfolio_weights(raw_weights, active_assets, max_stock_weight):
    """
    Project raw weights into the feasible portfolio simplex:
    - 0 <= stock_weight <= max_stock_weight
    - 0 <= fund_weight <= MAX_FUND_WEIGHT (0.40)
    - MIN_FUND_ALLOCATION (0.20) <= fund_total <= MAX_FUND_ALLOCATION (0.60)
    - sum(weights) == 1.0
    """
    w = np.maximum(np.asarray(raw_weights, dtype=float), 0.0)
    if w.sum() < 1e-8:
        w = np.ones_like(w) / len(w)
    else:
        w /= w.sum()

    stock_indices = [i for i, name in enumerate(active_assets) if name.startswith("STOCK_")]
    fund_indices = [i for i, name in enumerate(active_assets) if name.startswith("FUND_")]

    # Enforce fund allocation bounds [0.20, 0.60]
    fund_sum = float(w[fund_indices].sum()) if fund_indices else 0.0
    if fund_sum < MIN_FUND_ALLOCATION and fund_indices:
        deficit = MIN_FUND_ALLOCATION - fund_sum
        w[fund_indices] += deficit / len(fund_indices)
        if stock_indices and w[stock_indices].sum() > 1e-8:
            w[stock_indices] *= (1.0 - MIN_FUND_ALLOCATION) / w[stock_indices].sum()
    elif fund_sum > MAX_FUND_ALLOCATION and fund_indices:
        excess = fund_sum - MAX_FUND_ALLOCATION
        w[fund_indices] -= excess * (w[fund_indices] / max(fund_sum, 1e-8))
        if stock_indices and w[stock_indices].sum() > 1e-8:
            w[stock_indices] *= (1.0 - MAX_FUND_ALLOCATION) / w[stock_indices].sum()

    # Apply individual asset caps
    for i in stock_indices:
        w[i] = min(w[i], max_stock_weight)
    for i in fund_indices:
        w[i] = min(w[i], MAX_FUND_WEIGHT)

    w /= max(w.sum(), 1e-8)
    return w


def refine_candidate_differential_evolution(
    candidate_allocation,
    asset_returns,
    asset_index,
    sampled_month_indices,
    target,
    savings,
    monthly,
    max_loss,
    max_stock_weight,
    seed=42
):
    """
    Apply Differential Evolution refinement to fine-tune weights of promising candidate portfolios.
    Maximizes goal achievement probability and median wealth while penalizing 5th-percentile shortfall.
    """
    active_assets = list(candidate_allocation.keys())
    k = len(active_assets)
    if k < MIN_PORTFOLIO_ASSETS:
        return candidate_allocation, None

    init_weights = np.array([candidate_allocation[name] for name in active_assets], dtype=float)

    asset_sub_indices = [asset_index[name] for name in active_assets]
    sub_returns = asset_returns[:, asset_sub_indices]

    def objective(w_raw):
        w = project_portfolio_weights(w_raw, active_assets, max_stock_weight)
        # Vectorized portfolio simulation
        port_returns = sub_returns @ w
        sim_returns = port_returns[sampled_month_indices]
        wealth = np.full(sim_returns.shape[0], savings, dtype=float)
        for m in range(sim_returns.shape[1]):
            wealth *= (1.0 + sim_returns[:, m])
            wealth += monthly

        success_prob = float(np.mean(wealth >= target))
        med_wealth = float(np.median(wealth))
        p05_wealth = float(np.percentile(wealth, 5))
        p05_shortfall = max(0.0, target - p05_wealth)

        # Objective: Maximize success prob + median wealth, heavily penalize constraint violation
        penalty = 5.0 * max(0.0, p05_shortfall - max_loss) / (target + 1e-8)
        score = success_prob * 100.0 + (med_wealth / target) * 10.0 - penalty * 100.0
        return -score

    bounds = [
        (0.01, max_stock_weight if name.startswith("STOCK_") else MAX_FUND_WEIGHT)
        for name in active_assets
    ]

    try:
        # Differential Evolution global search
        res = differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=15,
            popsize=8,
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=False
        )
        refined_w = project_portfolio_weights(res.x, active_assets, max_stock_weight)
        refined_alloc = {name: float(refined_w[i]) for i, name in enumerate(active_assets) if refined_w[i] > 1e-6}
        # Re-normalize
        total_w = sum(refined_alloc.values())
        refined_alloc = {k: v / total_w for k, v in refined_alloc.items()}
        return refined_alloc, res
    except Exception:
        return candidate_allocation, None


def evaluate_portfolios(
    returns,
    stock_names,
    fund_names,
    target,
    savings,
    monthly,
    years,
    max_loss,
    risk_score,
    strategy="random",
    refine_top=10,
):
    """Generate candidates, simulate them, rank them, and save results."""

    if target <= 0:
        raise ValueError("Target must be greater than zero.")

    if savings < 0 or monthly < 0:
        raise ValueError("Savings and contributions cannot be negative.")

    if years <= 0:
        raise ValueError("Years must be greater than zero.")

    if max_loss < 0:
        raise ValueError("Maximum permitted shortfall cannot be negative.")

    months = years * 12
    max_stock_weight = get_stock_weight_limit(risk_score)

    asset_names = list(returns.columns)

    stock_names = [
        name for name in stock_names if name in asset_names
    ]

    fund_names = [
        name for name in fund_names if name in asset_names
    ]

    asset_returns = returns.to_numpy(dtype=float)

    asset_index = {
        name: index for index, name in enumerate(asset_names)
    }

    rng = np.random.default_rng(RANDOM_SEED)

    # Use identical sampled months for all candidates for fairer comparison.
    sampled_month_indices = rng.integers(
        0,
        len(returns),
        size=(NUM_SIMULATIONS, months),
    )

    candidate_rows = []
    weight_rows = []

    total_contributions = savings + monthly * months

    print("\nOPTIMIZATION SETTINGS")
    print(f"Target amount: {money(target)}")
    print(f"Current savings: {money(savings)}")
    print(f"Monthly contribution: {money(monthly)}")
    print(f"Horizon: {years} years")
    print(f"Total planned contributions: {money(total_contributions)}")
    print(f"Risk score: {risk_score}")
    print(f"Maximum individual stock weight: {max_stock_weight:.0%}")
    print(f"Maximum individual fund weight: {MAX_FUND_WEIGHT:.0%}")
    print(
        f"Mutual-fund allocation range: "
        f"{MIN_FUND_ALLOCATION:.0%}-{MAX_FUND_ALLOCATION:.0%}"
    )
    print(f"Maximum permitted 5th-percentile goal shortfall: {money(max_loss)}")
    print(f"Candidates: {NUM_CANDIDATES:,}")
    print(f"Simulations per candidate: {NUM_SIMULATIONS:,}")

    for candidate_id in range(1, NUM_CANDIDATES + 1):
        allocation = create_candidate_weights(
            stock_names,
            fund_names,
            rng,
            max_stock_weight,
        )

        weights = np.zeros(
            len(asset_names),
            dtype=float,
        )

        for name, weight in allocation.items():
            weights[asset_index[name]] = weight

        terminal_wealth = simulate_portfolio(
            asset_returns,
            weights,
            sampled_month_indices,
            savings,
            monthly,
        )

        expected_wealth = float(np.mean(terminal_wealth))
        median_wealth = float(np.median(terminal_wealth))
        p05_wealth = float(np.percentile(terminal_wealth, 5))

        success_probability = float(
            np.mean(terminal_wealth >= target)
        )

        p05_goal_shortfall = max(
            0.0,
            target - p05_wealth,
        )

        p05_capital_shortfall = max(
            0.0,
            total_contributions - p05_wealth,
        )

        meets_loss_limit = p05_goal_shortfall <= max_loss

        stock_total = sum(
            weight for name, weight in allocation.items()
            if name.startswith("STOCK_")
        )

        fund_total = sum(
            weight for name, weight in allocation.items()
            if name.startswith("FUND_")
        )

        candidate_rows.append({
            "candidate_id": candidate_id,
            "goal_success_probability": success_probability,
            "expected_terminal_wealth": expected_wealth,
            "median_terminal_wealth": median_wealth,
            "p05_terminal_wealth": p05_wealth,
            "p05_goal_shortfall": p05_goal_shortfall,
            "p05_capital_shortfall_vs_contributions": p05_capital_shortfall,
            "meets_loss_limit": meets_loss_limit,
            "number_of_holdings": len(allocation),
            "stock_allocation": stock_total,
            "fund_allocation": fund_total,
            "largest_holding_weight": max(allocation.values()),
        })

        weight_row = {"candidate_id": candidate_id}

        for name in asset_names:
            weight_row[name] = allocation.get(name, 0.0)

        weight_rows.append(weight_row)

    # Initial Random Population
    candidates = pd.DataFrame(candidate_rows)
    candidate_weights = pd.DataFrame(weight_rows)

    # -------------------------------------------------------------
    # OPTIONAL HYBRID REFINEMENT: DIFFERENTIAL EVOLUTION
    # -------------------------------------------------------------
    if strategy in ("hybrid", "differential_evolution"):
        print(f"\n[Hybrid Optimization] Refining Top {refine_top} Candidates with Differential Evolution...")
        t_de_start = time.time()
        
        # Identify top promising candidates based on goal probability & median wealth
        pre_ranked = candidates.sort_values(
            ["goal_success_probability", "median_terminal_wealth"],
            ascending=[False, False]
        ).head(refine_top)
        
        refined_candidates = []
        refined_weights = []
        
        for idx, row in pre_ranked.iterrows():
            cand_id = int(row["candidate_id"])
            c_weights_row = candidate_weights.loc[candidate_weights["candidate_id"] == cand_id].iloc[0]
            c_alloc = {name: float(c_weights_row[name]) for name in asset_names if float(c_weights_row[name]) > 1e-6}
            
            ref_alloc, de_res = refine_candidate_differential_evolution(
                candidate_allocation=c_alloc,
                asset_returns=asset_returns,
                asset_index=asset_index,
                sampled_month_indices=sampled_month_indices,
                target=target,
                savings=savings,
                monthly=monthly,
                max_loss=max_loss,
                max_stock_weight=max_stock_weight,
                seed=RANDOM_SEED + cand_id
            )
            
            # Re-simulate refined candidate
            ref_w = np.zeros(len(asset_names), dtype=float)
            for name, weight in ref_alloc.items():
                ref_w[asset_index[name]] = weight
                
            ref_wealth = simulate_portfolio(asset_returns, ref_w, sampled_month_indices, savings, monthly)
            ref_expected = float(np.mean(ref_wealth))
            ref_median = float(np.median(ref_wealth))
            ref_p05 = float(np.percentile(ref_wealth, 5))
            ref_prob = float(np.mean(ref_wealth >= target))
            ref_shortfall = max(0.0, target - ref_p05)
            ref_cap_shortfall = max(0.0, total_contributions - ref_p05)
            ref_meets = ref_shortfall <= max_loss
            
            new_id = int(100000 + cand_id)
            stock_tot = sum(w for k, w in ref_alloc.items() if k.startswith("STOCK_"))
            fund_tot = sum(w for k, w in ref_alloc.items() if k.startswith("FUND_"))
            
            refined_candidates.append({
                "candidate_id": new_id,
                "goal_success_probability": ref_prob,
                "expected_terminal_wealth": ref_expected,
                "median_terminal_wealth": ref_median,
                "p05_terminal_wealth": ref_p05,
                "p05_goal_shortfall": ref_shortfall,
                "p05_capital_shortfall_vs_contributions": ref_cap_shortfall,
                "meets_loss_limit": ref_meets,
                "number_of_holdings": len(ref_alloc),
                "stock_allocation": stock_tot,
                "fund_allocation": fund_tot,
                "largest_holding_weight": max(ref_alloc.values()),
                "optimization_method": "differential_evolution_refined"
            })
            
            ref_w_row = {"candidate_id": new_id}
            for name in asset_names:
                ref_w_row[name] = ref_alloc.get(name, 0.0)
            refined_weights.append(ref_w_row)
            
        de_elapsed = time.time() - t_de_start
        print(f"  Differential Evolution refinement complete in {de_elapsed:.2f}s across {len(refined_candidates)} candidates.")
        
        # Merge refined candidates into candidate pool
        candidates = pd.concat([candidates, pd.DataFrame(refined_candidates)], ignore_index=True)
        candidate_weights = pd.concat([candidate_weights, pd.DataFrame(refined_weights)], ignore_index=True)

    candidates["constraint_excess"] = (
        candidates["p05_goal_shortfall"] - max_loss
    ).clip(lower=0.0)

    candidates["meets_goal_constraint"] = (
        candidates["constraint_excess"] <= 1e-9
    )

    feasible = candidates[
        candidates["meets_goal_constraint"]
    ]

    if not feasible.empty:
        ranked = feasible.sort_values(
            [
                "goal_success_probability",
                "median_terminal_wealth",
            ],
            ascending=[False, False],
        )

        selection_note = (
            "Selected from candidates satisfying the shortfall constraint."
        )

    else:
        ranked = candidates.sort_values(
            [
                "constraint_excess",
                "goal_success_probability",
                "median_terminal_wealth",
            ],
            ascending=[True, False, False],
        )

        selection_note = (
            "No feasible candidate was found in this random search. "
            "The selected portfolio minimizes constraint excess first."
        )

    best = ranked.iloc[0]
    best_id = int(best["candidate_id"])

    best_weights_row = candidate_weights.loc[
        candidate_weights["candidate_id"] == best_id
    ].iloc[0]

    allocation = {
        name: float(best_weights_row[name])
        for name in asset_names
        if float(best_weights_row[name]) > 1e-8
    }

    allocation_df = pd.DataFrame([
        {
            "asset": name,
            "weight": weight,
            "weight_percent": weight * 100,
        }
        for name, weight in allocation.items()
    ]).sort_values("weight", ascending=False)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.sort_values(
        [
            "meets_goal_constraint",
            "goal_success_probability",
            "median_terminal_wealth",
        ],
        ascending=[False, False, False],
    ).to_csv(
        PROCESSED_DIR / "goal_optimizer_candidates.csv",
        index=False,
    )

    candidate_weights.to_csv(
        PROCESSED_DIR / "goal_optimizer_candidate_weights.csv",
        index=False,
    )

    allocation_df.to_csv(
        PROCESSED_DIR / "goal_optimizer_allocation.csv",
        index=False,
    )

    closest = candidates.sort_values(
        [
            "constraint_excess",
            "p05_goal_shortfall",
            "goal_success_probability",
        ],
        ascending=[True, True, False],
    ).head(20)

    closest.to_csv(
        PROCESSED_DIR / "goal_optimizer_closest_candidates.csv",
        index=False,
    )

    print("\nOPTIMIZATION RESULTS")
    print(f"Selection note: {selection_note}")
    print(f"Selected candidate ID: {best_id}")
    print(
        "Simulated goal-achievement probability: "
        f"{best['goal_success_probability']:.2%}"
    )
    print(f"Expected terminal wealth: {money(best['expected_terminal_wealth'])}")
    print(f"Median terminal wealth: {money(best['median_terminal_wealth'])}")
    print(f"5th-percentile terminal wealth: {money(best['p05_terminal_wealth'])}")
    print(f"5th-percentile shortfall from goal: {money(best['p05_goal_shortfall'])}")
    print(
        "5th-percentile shortfall vs contributions: "
        f"{money(best['p05_capital_shortfall_vs_contributions'])}"
    )
    print(
        "Goal-shortfall constraint: "
        f"{'SATISFIED' if best['meets_goal_constraint'] else 'NOT SATISFIED'}"
    )

    print("\nSELECTED CANDIDATE ALLOCATION")

    for _, row in allocation_df.iterrows():
        print(
            f"{row['asset']}: "
            f"{row['weight_percent']:.2f}%"
        )

    print("\nDIVERSIFICATION SUMMARY")
    print(f"Number of holdings: {len(allocation_df)}")
    print(
        "Largest individual allocation: "
        f"{allocation_df['weight'].max():.2%}"
    )

    print(
        "Total direct-stock allocation: "
        f"{allocation_df.loc[allocation_df['asset'].str.startswith('STOCK_'), 'weight'].sum():.2%}"
    )

    print(
        "Total mutual-fund allocation: "
        f"{allocation_df.loc[allocation_df['asset'].str.startswith('FUND_'), 'weight'].sum():.2%}"
    )

    print(
        "Underlying mutual-fund holdings are not checked for overlap "
        "with direct stock positions."
    )

    print("\nOUTPUT FILES")

    print(
        PROCESSED_DIR / "goal_optimizer_candidates.csv"
    )
    print(
        PROCESSED_DIR / "goal_optimizer_candidate_weights.csv"
    )
    print(
        PROCESSED_DIR / "goal_optimizer_allocation.csv"
    )
    print(
        PROCESSED_DIR / "goal_optimizer_closest_candidates.csv"
    )

    print(
        "\nReminder: historical simulations do not guarantee future performance."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize a portfolio against a target wealth goal."
    )

    parser.add_argument(
        "--target",
        type=float,
        required=True,
        help="Target wealth in rupees.",
    )

    parser.add_argument(
        "--savings",
        type=float,
        required=True,
        help="Current savings in rupees.",
    )

    parser.add_argument(
        "--monthly",
        type=float,
        required=True,
        help="Monthly contribution in rupees.",
    )

    parser.add_argument(
        "--years",
        type=int,
        required=True,
        help="Investment horizon in years.",
    )

    parser.add_argument(
        "--max-loss",
        type=float,
        required=True,
        help=(
            "Maximum permitted 5th-percentile shortfall "
            "from the target, in rupees."
        ),
    )

    parser.add_argument(
        "--risk-score",
        type=int,
        required=True,
        help="Risk score from 1 (lower) to 10 (higher).",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "hybrid", "differential_evolution"],
        help="Portfolio search strategy: random (baseline) or hybrid (random + differential evolution refinement).",
    )

    parser.add_argument(
        "--refine-top",
        type=int,
        default=10,
        help="Number of promising candidates to refine using Differential Evolution in hybrid mode.",
    )

    args = parser.parse_args()

    returns, stock_names, fund_names = load_data()

    get_gsec_context()

    evaluate_portfolios(
        returns=returns,
        stock_names=stock_names,
        fund_names=fund_names,
        target=args.target,
        savings=args.savings,
        monthly=args.monthly,
        years=args.years,
        max_loss=args.max_loss,
        risk_score=args.risk_score,
        strategy=args.strategy,
        refine_top=args.refine_top,
    )


if __name__ == "__main__":
    main()
