"""Kiểm thử phần mở rộng: tham số ghi đè, Stackelberg, hiệu chỉnh lịch sử. Chạy: pytest -q"""
import csv

import pytest

import calibration
import run_montecarlo_historical as mc
import run_stackelberg as st
from model import DeFiContagionModel


def test_default_unchanged_after_adding_overrides():
    m = DeFiContagionModel(seed=42)
    for _ in range(15):
        m.step()
    assert m.total_liquidations == 185 and round(m.bad_debt) == 291012


def test_bonus_override_applied_and_lt_changes_debt():
    m = DeFiContagionModel(seed=1, liquidation_bonuses={"WETH": 0.2})
    assert m.liquidation_bonuses["WETH"] == 0.2 and m.liquidation_bonuses["WBTC"] == 0.08
    base = sum(b.debt for b in DeFiContagionModel(seed=1).borrowers())
    hi = sum(b.debt for b in DeFiContagionModel(seed=1, liquidation_thresholds={"WETH": 0.9}).borrowers())
    assert hi > base   # LT cao hơn -> cùng HF0 -> nợ ban đầu lớn hơn


def test_stackelberg_cell_reproducible_and_bounded():
    a = st.run_one(3, 0.5, 1.0)
    b = st.run_one(3, 0.5, 1.0)
    assert a == b
    assert 0.0 <= a["unresolved"] <= 1.0 and a["bonus_paid"] >= 0 and a["bad_debt"] >= 0


def test_no_liquidation_when_bonus_too_low_means_zero_bonus_paid():
    r = st.run_one(0, 1.0, 0.1)
    assert r["liquidations"] == 0 and r["bonus_paid"] == 0 and r["unresolved"] > 0


def test_best_response_respects_constraint():
    grid = st.build_grid(n_seeds=2)
    assert len(grid) == len(st.KAPPAS) * len(st.BONUS_SCALES)
    tab = st.best_response_table(grid)
    for _, r in tab.dropna(subset=["kappa*"]).iterrows():
        assert r["tồn đọng"] <= r["RHO"] + 1e-12
        assert r["tham gia"] >= r["PART"] - 1e-12


def test_participation_constraint_excludes_no_liquidation_cells():
    grid = st.build_grid(n_seeds=2)
    low = grid[(grid.kappa == 1.0) & (grid.bonus_scale == 0.25)].iloc[0]
    assert low["participation"] < 0.1 and low["bad_debt"] == 0   # nợ xấu = 0 chỉ vì không ai thanh lý
    assert (1.0, 0.25) not in set(zip(st.feasible_cells(grid, 0.25, st.PART_MIN).kappa,
                                      st.feasible_cells(grid, 0.25, st.PART_MIN).bonus_scale))
    base = grid[(grid.kappa == 0.5) & (grid.bonus_scale == 1.0)].iloc[0]
    assert base["participation"] == 1.0 and base["bad_debt_vs_base"] == 0.0


def test_drops_and_var_es():
    d = calibration.h_day_drops([100, 90, 80, 100], 1)
    assert d == pytest.approx([0.1, 1 - 80 / 90, -0.25])
    var, es = calibration.var_es(list(range(100)), 0.95)
    assert var == pytest.approx(94.05) and es == pytest.approx(97.0)
    assert calibration.var_es([0.0] * 19 + [10.0], 0.95)[1] == 10.0   # ES không bị pha loãng khi VaR = 0


def test_montecarlo_pipeline_on_tiny_fixture(tmp_path):
    path = tmp_path / "p.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "close"])
        for i, p in enumerate([100, 95, 70, 72, 60, 65, 50, 55, 52, 58, 40, 45]):
            w.writerow([f"2020-01-{i + 1:02d}", p])
    drops = calibration.h_day_drops(calibration.load_prices(str(path)), 3)
    import random
    shocks = mc.sample_shocks(drops, 5, random.Random(0))
    assert all(mc.MIN_DROP <= s <= 0.95 for s in shocks)
    r = mc.run_shock(shocks[0], 1, calibration.PUBLISHED_PARAMS)
    assert r["bad_debt"] >= 0 and isinstance(r["contagion"], bool)


def test_montecarlo_other_assets_and_episodes():
    for asset in ("WBTC", "LINK"):
        r = mc.run_shock(0.4, 5, {}, asset)
        assert r["bad_debt"] >= 0 and isinstance(r["contagion"], bool)
    dates = [f"2020-01-{i + 1:02d}" for i in range(12)]
    prices = [100, 95, 70, 72, 60, 65, 50, 55, 52, 58, 40, 45]
    ep = calibration.worst_episodes(dates, prices, 3, k=2)
    assert len(ep) == 2 and ep[0][2] >= ep[1][2]
    assert abs(int(ep[0][0][-2:]) - int(ep[1][0][-2:])) >= 3   # không chồng lấn


# ---------------------------------------------------------------- hai kịch bản sốc (scenarios.py)
import scenarios as sc


def _initial_state(model):
    return [(b.unique_id, b.group_id, round(b.debt, 6), tuple(sorted(b.collaterals.items())))
            for b in model.borrowers()]


def test_both_scenarios_start_from_same_initial_state():
    a = sc.build_model(sc.SCENARIOS["single"])
    b = sc.build_model(sc.SCENARIOS["multi"])
    assert _initial_state(a) == _initial_state(b)
    assert a.oracle_prices == b.oracle_prices and a.amm_pools["WETH"].k == b.amm_pools["WETH"].k


def test_single_scenario_matches_original_model():
    m, df = sc.run_scenario(sc.SCENARIOS["single"], 15, seed=42)
    assert m.total_liquidations == 185 and round(m.bad_debt) == 291012
    assert int(df["liquidation_events"].sum()) == 185


def test_multi_shock_applied_simultaneously_at_tick1():
    m = sc.build_model(sc.SCENARIOS["multi"])
    m.step()                                   # tick 1
    # Oracle ở chế độ AMM nên giá = giá bể sau sốc; kiểm tra giá mục tiêu ngoại sinh (đúng mức sốc)
    for a, pct in sc.SCENARIOS["multi"].items():
        assert m.external_target_prices[a] == pytest.approx(m.baseline_prices[a] * (1 - pct))
    assert m.external_target_prices["USDC"] == 1.0
    assert m.current_tick == 1


def test_single_default_shocks_only_weth():
    m = sc.build_model(sc.SCENARIOS["single"])
    m.step()
    assert m.external_target_prices["WBTC"] == m.baseline_prices["WBTC"]
    assert m.external_target_prices["LINK"] == m.baseline_prices["LINK"]
    assert m.external_target_prices["WETH"] == pytest.approx(3000 * 0.65)


def test_scenarios_independent_of_run_order():
    s1, _ = sc.run_scenario(sc.SCENARIOS["single"], 15, seed=42)
    m1, _ = sc.run_scenario(sc.SCENARIOS["multi"], 15, seed=42)
    s2, _ = sc.run_scenario(sc.SCENARIOS["single"], 15, seed=42)
    assert s1.total_liquidations == s2.total_liquidations and s1.bad_debt == s2.bad_debt


def test_tick_table_consistency():
    m, df = sc.run_scenario(sc.SCENARIOS["single"], 15, seed=42)
    assert len(df) == 16 and df["tick"].tolist() == list(range(16))
    assert df["liquidation_events"].sum() == len(m.liquidation_events)
    assert df["cum_liquidated_borrowers"].iloc[-1] == len({e["borrower_id"] for e in m.liquidation_events})
    assert (df["safe"] + df["at_risk"] + df["liquidatable"] == 60).all()
    assert df["liquidated_value_usd"].sum() > df["debt_covered_usd"].sum()   # thế chấp tịch thu = nợ x (1 + thưởng)


def test_shock_pcts_validation():
    with pytest.raises(ValueError):
        DeFiContagionModel(shock_pcts={"WETH": 1.2})
    with pytest.raises(ValueError):
        DeFiContagionModel(shock_pcts={"DOGE": 0.1})
    with pytest.raises(ValueError):
        DeFiContagionModel(shock_pcts={})
