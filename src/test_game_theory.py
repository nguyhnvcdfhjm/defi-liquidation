"""Kiểm thử module lý thuyết trò chơi (game_theory.py). Chạy: pytest -q test_game_theory.py"""
import pytest

from game_theory import (GameConfig, GameTheoryModel, best_response, expected_cost,
                         predicted_hf, rollout, sandbox)
from model import DeFiContagionModel


def _run(seed=42, game=None, ticks=15, **kw):
    m = GameTheoryModel(seed=seed, game=game, **kw)
    cash0 = {b.unique_id: b.cash_usdc for b in m.borrowers()}
    for _ in range(ticks):
        m.step()
    return m, sum(cash0[b.unique_id] - b.cash_usdc for b in m.borrowers())


def _state(m):
    return (
        {a: (p.reserve_token, p.reserve_usdc) for a, p in m.amm_pools.items()},
        dict(m.oracle_prices), dict(m.external_target_prices),
        [(b.unique_id, dict(b.collaterals), b.debt, b.cash_usdc, b.health_factor,
          b.is_liquidated, b.bad_debt_recorded) for b in m.borrowers()],
        (m.total_liquidations, m.stalled_liquidations, m.bad_debt, len(m.liquidation_events), m.current_tick),
        [(l.total_profit, l.priority_fees_paid) for l in m.liquidators],
        m._sim_rng.getstate(),
    )


def _after_shock(seed=42, n_liquidators=1, **cfg):
    m = GameTheoryModel(seed=seed, n_liquidators=n_liquidators, game=GameConfig(**cfg))
    m.current_tick = 1
    m.inject_shock()
    for b in m.borrowers():
        b.step()
    return m


def test_game_none_reproduces_baseline_exactly():
    m, _ = _run(game=None)
    assert m.total_liquidations == 185 and round(m.bad_debt) == 291012
    base = DeFiContagionModel(seed=42)
    for _ in range(15):
        base.step()
    pd_a = m.datacollector.get_model_vars_dataframe()
    pd_b = base.datacollector.get_model_vars_dataframe()
    assert pd_a.equals(pd_b)


def test_sandbox_restores_every_field_including_rng():
    m = _after_shock(n_liquidators=3, players="all")
    before = _state(m)
    uid = m.borrowers()[0].unique_id
    rollout(m, {uid: 100.0}, 4)
    assert _state(m) == before
    with sandbox(m):
        m.liquidator.step()
    assert _state(m) == before


def test_rollout_matches_real_mechanics():
    m = _after_shock()
    tr = rollout(m, {}, 1).trough
    expect = dict(m.oracle_prices)
    m.liquidator.step()
    m.arbitrageur.step()
    for a in ("WETH", "WBTC", "LINK"):
        expect[a] = min(expect[a], m.oracle_prices[a])
    assert tr == pytest.approx(expect)


def test_high_cost_nobody_repays():
    m, repaid = _run(game=GameConfig(players="all", phi=10.0))
    assert repaid == 0.0
    assert all(l["repaid_usd"] == 0.0 for l in m.game.log)


def test_zero_cost_at_risk_players_repay_to_cap():
    m = _after_shock(players="all", phi=0.0)
    P = m.game.players(m)
    m.game.play(m)
    rep = m.game.log[-1]
    assert rep["n_repaying"] > 0 and rep["n_players"] == len(P)
    for b in P:
        assert b.cash_usdc >= -1e-9 and b.debt >= 0


def test_budget_constraints_whole_run():
    m0 = GameTheoryModel(seed=42, game=GameConfig(players="all"))
    budget = sum(b.cash_usdc for b in m0.borrowers())
    m, repaid = _run(game=GameConfig(players="all"))
    for b in m.borrowers():
        assert b.cash_usdc >= -1e-6 and b.debt >= 0
    assert 0 < repaid <= budget + 1e-6   # tiền trả không vượt tổng ngân sách ban đầu


def test_deterministic_given_seed():
    a, ra = _run(seed=7, game=GameConfig(players="all"))
    b, rb = _run(seed=7, game=GameConfig(players="all"))
    assert (a.total_liquidations, a.bad_debt, ra) == (b.total_liquidations, b.bad_debt, rb)


def test_players_active_matches_baseline_active_set():
    base = DeFiContagionModel(seed=42)
    n_active = sum(b.risk_profile == "active" for b in base.borrowers())
    m = GameTheoryModel(seed=42, game=GameConfig(players="active"))
    assert sum(b.is_player for b in m.borrowers()) == n_active
    assert sum(not b.is_player for b in m.borrowers()) == 60 - n_active


def test_best_response_zero_when_safe_and_positive_when_marginal():
    m = _after_shock(players="all", phi=0.01)
    cfg = m.game_cfg
    b = next(b for b in m.borrowers() if b.debt > 0)
    calm = dict(m.baseline_prices)                       # giá chưa sốc, không có cú thác
    b.debt = b.market_value() * 0.30                     # HF rất cao
    assert best_response(m, cfg, b, calm) == 0.0
    b.debt = sum(q * calm[a] * m.liquidation_thresholds[a] for a, q in b.collaterals.items()) / 0.97
    assert predicted_hf(m, b, calm, 0.0) < 1.0           # HF dự báo < 1 nhưng chữa được bằng <= 20% nợ
    assert best_response(m, cfg, b, calm) > 0.0


def test_expected_cost_nonnegative_and_repay_cost_linear_when_safe():
    m = _after_shock(players="all", phi=0.05)
    cfg = m.game_cfg
    b = m.borrowers()[0]
    b.debt = b.market_value() * 0.1
    calm = dict(m.baseline_prices)
    c1, c2 = expected_cost(m, cfg, b, calm, 0.05), expected_cost(m, cfg, b, calm, 0.10)
    assert 0 <= c1 < c2 and c2 == pytest.approx(2 * c1, rel=1e-6)


def test_nash_epsilon_is_small():
    m, _ = _run(game=GameConfig(players="all"))
    assert max(l["epsilon"] for l in m.game.log) < 0.01   # hối tiếc < 1% nợ


def test_nash_all_lowers_bad_debt_every_seed():
    for s in range(5):
        base, _ = _run(seed=s, game=None)
        nash, _ = _run(seed=s, game=GameConfig(players="all"))
        assert nash.bad_debt < base.bad_debt


def test_invalid_config_rejected():
    for kw in ({"mode": "x"}, {"players": "x"}, {"n_actions": 1}, {"phi": -1.0}, {"tau": 0.0}):
        with pytest.raises(ValueError):
            GameConfig(**kw)
