"""test_simulation.py - 4 bài kiểm thử chấp nhận (+ vài kiểm thử phụ). Chạy: pytest -v test_simulation.py"""
import mesa
import pytest

from amm import AMMPool
from agents import BorrowerAgent
from model import DeFiContagionModel


class _StubModel(mesa.Model):
    """Mô hình tối giản chỉ chứa giá oracle và LT để kiểm thử BorrowerAgent."""

    def __init__(self):
        super().__init__()
        self.oracle_prices = {"WETH": 3000.0, "WBTC": 60000.0, "LINK": 15.0, "USDC": 1.0}
        self.liquidation_thresholds = {"WETH": 0.82, "WBTC": 0.75, "LINK": 0.65}


def test_01_amm_invariant():
    pool = AMMPool("WBTC", 100.0, 100_000.0, fee_rate=0.003)
    out, p_post = pool.swap_token_for_usdc(10.0)
    assert out == pytest.approx(9066.11, abs=0.01)
    assert p_post == pytest.approx(826.67, abs=0.01)
    assert p_post < 1000.0
    assert pool.reserve_token * pool.reserve_usdc >= 10_000_000


def test_02_health_factor_calculation():
    b = BorrowerAgent(_StubModel(), {"WETH": 2.0, "WBTC": 0.1}, debt=7000.0, group_id=0)
    assert b.calculate_health_factor() == pytest.approx(9420 / 7000)
    assert b.calculate_health_factor() == pytest.approx(1.345714, abs=1e-6)


def test_03_contagion_cascade_trigger():
    m = DeFiContagionModel(n_borrowers=60, shock_pct=0.35, seed=42)
    snap = {}
    for _ in range(6):
        m.step()
        snap[m.current_tick] = dict(m.oracle_prices)
    assert snap[3]["WBTC"] < 50_000
    # Nhóm 2 (group_id=1, không giữ WETH) có ít nhất 1 vụ thanh lý tính đến tick 3
    assert any(e["group_id"] == 1 and e["tick"] <= 3 for e in m.liquidation_events)
    assert snap[6]["LINK"] < 10
    assert any(e["group_id"] == 2 and e["tick"] <= 6 for e in m.liquidation_events)


def test_04_bad_debt_under_liquidity_crunch():
    m = DeFiContagionModel(n_borrowers=60, shock_pct=0.65, amm_depth_multiplier=0.1, seed=42)
    m.step()
    bad_t1 = m.bad_debt
    assert bad_t1 > 0
    m.step()
    assert m.bad_debt == bad_t1  # không đếm trùng lặp


# ----------------------------- kiểm thử phụ -----------------------------
def test_collateral_cap_scales_debt_proportionally():
    """Số dư thế chấp nhỏ hơn nhu cầu -> tịch thu toàn bộ và co giảm nợ trả thay theo tỷ lệ."""
    m = DeFiContagionModel(n_borrowers=3, seed=1)
    b = m.borrowers()[0]
    asset = next(iter(b.collaterals))
    b.collaterals = {asset: 0.001}
    price, bonus = m.oracle_prices[asset], m.liquidation_bonuses[asset]
    plan = m.liquidator._plan(b, asset, desired_debt=1e9)
    seize, debt_cov, _ = plan
    assert seize == pytest.approx(0.001)
    assert debt_cov == pytest.approx(0.001 * price / (1 + bonus))


def test_shock_reprices_amm_and_is_idempotent():
    m = DeFiContagionModel(seed=1)
    m.inject_shock()
    m.inject_shock()
    assert m.oracle_prices["WETH"] == pytest.approx(1950.0)
    assert m.amm_spot_prices["WETH"] == pytest.approx(1950.0)
    assert m.external_target_prices["WETH"] == pytest.approx(1950.0)


def test_arbitrageur_never_restores_pre_shock_price():
    m = DeFiContagionModel(n_borrowers=30, seed=1)
    for _ in range(30):
        m.step()
    assert m.amm_spot_prices["WETH"] <= 1950.0 * 1.0001


def test_depth_multiplier_preserves_spot_price():
    m = DeFiContagionModel(amm_depth_multiplier=0.1, seed=1)
    assert m.amm_spot_prices["WBTC"] == pytest.approx(60000.0)
    assert m.amm_pools["WBTC"].k == pytest.approx(0.01 * 15.0 * 900_000.0)


def test_determinism_same_seed():
    def run(seed):
        m = DeFiContagionModel(seed=seed)
        for _ in range(10):
            m.step()
        return (m.total_liquidations, round(m.bad_debt, 6), round(m.oracle_prices["WBTC"], 6))

    assert run(7) == run(7)


def test_datacollector_schema():
    m = DeFiContagionModel(seed=1)
    for _ in range(3):
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    assert len(df) == 4
    for col in ["Price_WETH", "Price_WBTC", "Price_LINK", "Spot_WETH", "Spot_WBTC", "Spot_LINK",
                "Total_Liquidations", "Bad_Debt", "Insolvent_Count", "Avg_HF"]:
        assert col in df.columns
    required = {"tick", "borrower_id", "group_id", "seized_asset", "seized_amount",
                "debt_covered", "profit", "new_spot_price"}  # schema theo đặc tả (có thể có thêm trường)
    assert required <= set(m.liquidation_events[0])


def test_invalid_params():
    with pytest.raises(ValueError):
        DeFiContagionModel(shock_asset="DOGE")


# ------------------- kiểm thử đối chứng: bật/tắt kênh lây lan -------------------
def _indirect_events(model):
    # Nhóm 2 và Nhóm 3 (group_id 1, 2) không giữ WETH -> thanh lý ở đây là do lây lan.
    return [e for e in model.liquidation_events if e["group_id"] in (1, 2)]


def test_channel_on_causes_indirect_liquidations():
    m = DeFiContagionModel(seed=42, oracle_mode="amm")
    for _ in range(15):
        m.step()
    assert len(_indirect_events(m)) > 0


def test_channel_off_removes_indirect_liquidations():
    m = DeFiContagionModel(seed=42, oracle_mode="external")
    for _ in range(15):
        m.step()
    assert len(_indirect_events(m)) == 0 and m.bad_debt == 0


def test_invalid_oracle_mode():
    with pytest.raises(ValueError):
        DeFiContagionModel(oracle_mode="x")


# ------------------- kiểm thử cạnh tranh gas (đấu giá kiểu Anh) -------------------
def test_auction_winner_pays_second_value_and_profit_is_cost_gap():
    from agents import english_auction
    # gộp = 100; chi phí [10, 30] -> giá trị [90, 70]: bot 0 thắng, phí ưu tiên = 70, lợi nhuận = 90 - 70 = 20 = 30 - 10
    winner, fee, profit = english_auction(100.0, [10.0, 30.0])
    assert winner == 0 and fee == pytest.approx(70.0) and profit == pytest.approx(20.0)


def test_auction_single_bidder_has_no_priority_fee():
    from agents import english_auction
    winner, fee, profit = english_auction(100.0, [20.0])
    assert winner == 0 and fee == 0.0 and profit == pytest.approx(80.0)


def test_auction_no_one_liquidates_when_unprofitable():
    from agents import english_auction
    winner, fee, profit = english_auction(5.0, [10.0, 30.0])
    assert winner is None and fee == 0.0 and profit == 0.0


def test_auction_expected_profit_matches_theory():
    """Kỳ vọng lợi nhuận người thắng = g/(n+1) khi G lớn (chi phí ~ U(0.5g, 1.5g))."""
    import random
    from agents import english_auction
    rng, g = random.Random(0), 20.0
    for n in (2, 5):
        profits = [english_auction(1000.0, [g * rng.uniform(0.5, 1.5) for _ in range(n)])[2]
                   for _ in range(20000)]
        assert sum(profits) / len(profits) == pytest.approx(g / (n + 1), abs=0.15)


def test_competition_does_not_change_single_bot_default():
    """n_liquidators=1 phải cho kết quả y hệt bản gốc (số seed 42, 15 tick)."""
    a = DeFiContagionModel(seed=42)
    b = DeFiContagionModel(seed=42, n_liquidators=1)
    for _ in range(15):
        a.step()
        b.step()
    assert a.total_liquidations == b.total_liquidations and a.bad_debt == b.bad_debt


def test_many_bots_events_carry_auction_fields():
    m = DeFiContagionModel(seed=42, n_liquidators=5)
    for _ in range(5):
        m.step()
    ev = m.liquidation_events
    assert ev and all(e["n_bidders"] == 5 and e["priority_fee"] >= 0 for e in ev)
    assert len({e["liquidator_id"] for e in ev}) > 1  # nhiều bot khác nhau thắng
    assert sum(b.total_profit for b in m.liquidators) == pytest.approx(sum(e["profit"] for e in ev))


def test_invalid_n_liquidators():
    with pytest.raises(ValueError):
        DeFiContagionModel(n_liquidators=0)
