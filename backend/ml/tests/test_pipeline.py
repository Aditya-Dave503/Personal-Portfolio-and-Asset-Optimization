"""End-to-end integration tests for ml.pipeline."""

import sys
from pathlib import Path
import pytest

from ml import pipeline


def test_run_pipeline_end_to_end_with_risk_profiler(synthetic_data_dir: Path, isolated_model_dir: Path):
    portfolio = pipeline.run_pipeline(
        investment_amount=500000.0,
        age=32,
        investment_horizon_years=12,
        savings_to_income_ratio=0.6,
        max_loss_tolerance_pct=0.20,
        optimization_mode="black_litterman",
        retrain=True,
        data_dir_str=str(synthetic_data_dir),
    )

    assert portfolio["investment_amount"] == 500000.0
    assert 1 <= portfolio["risk_score"] <= 10
    assert len(portfolio["allocations"]) == 4
    assert sum(a["weight_percent"] for a in portfolio["allocations"]) == pytest.approx(100.0, abs=0.1)


def test_run_pipeline_explicit_risk_score_and_target_amount(
    synthetic_data_dir: Path, isolated_model_dir: Path
):
    portfolio = pipeline.run_pipeline(
        investment_amount=100000.0,
        target_amount=1200000.0,
        explicit_risk_score=8,
        optimization_mode="max_sharpe",
        retrain=True,
        data_dir_str=str(synthetic_data_dir),
    )

    assert portfolio["investment_amount"] == 1200000.0
    assert portfolio["risk_score"] == 8
    assert portfolio["risk_profile_category"] == "Aggressive Growth"


def test_pipeline_cli_main(
    synthetic_data_dir: Path, isolated_model_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline.py",
            "--amount",
            "300000",
            "--risk-score",
            "5",
            "--mode",
            "min_variance",
            "--train",
            "--data-dir",
            str(synthetic_data_dir),
        ],
    )
    pipeline.main()
    captured = capsys.readouterr()
    assert "OPTIMAL INVESTMENT ADVISORY ALLOCATION" in captured.out
    assert "300,000.00" in captured.out

