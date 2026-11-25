"""
scenarios.py - Hai kịch bản cú sốc giá để so sánh độ lây lan rủi ro thanh lý.

  single : sốc ĐƠN tài sản  - WETH giảm 35% tại tick 1, các tài sản khác giữ nguyên.
  multi  : sốc ĐA tài sản   - WETH -22%, WBTC -15%, LINK -15% ĐỒNG THỜI tại tick 1 (tổng các tỷ lệ = 52%).
            (Mức ban đầu 15/10/10, tổng 35%, không gây thanh lý nào ở seed 42 nên đã nâng lên 22/15/15.)
  USDC (đồng ổn định) không bị sốc ở cả hai.

Hai kịch bản dùng CÙNG mô hình, công thức HF, LT, quy tắc thanh lý; chỉ khác tham số shock_pcts.
Quần thể người vay chỉ phụ thuộc seed nên hai kịch bản bắt đầu từ cùng một trạng thái (xem test).
Mỗi lần chạy tạo một mô hình mới, không chia sẻ trạng thái giữa các kịch bản.

Lưu ý khi diễn giải: "tổng các tỷ lệ" là tổng các tỷ lệ sốc đặt vào, không phải tổn thất danh mục. Tổn thất
danh mục phụ thuộc tỷ trọng từng tài sản trong thế chấp, nên hai kịch bản không cùng giá trị thế chấp bị mất.
"""
from typing import Dict, List, Tuple

import pandas as pd

from model import DeFiContagionModel
from agents import BorrowerAgent

SCENARIOS: Dict[str, Dict[str, float]] = {
    "single": {"WETH": 0.35},
    "multi": {"WETH": 0.22, "WBTC": 0.15, "LINK": 0.15},
}
SCENARIO_LABELS = {"single": "Sốc đơn tài sản", "multi": "Sốc đa tài sản"}
ASSETS = ("WETH", "WBTC", "LINK")
RISK_HF = 1.15   # HF < 1,15: vùng rủi ro (cùng ngưỡng người vay 'active' bắt đầu tự trả nợ)

DEFAULT_PARAMS = dict(n_borrowers=60, amm_depth_multiplier=1.0, close_factor=0.50, seed=42,
                      oracle_mode="amm", n_liquidators=1)


def build_model(shocks: Dict[str, float], **params) -> DeFiContagionModel:
    """Mô hình mới cho một kịch bản (tham số thiếu lấy từ DEFAULT_PARAMS)."""
    p = {**DEFAULT_PARAMS, **{k: v for k, v in params.items() if k in DEFAULT_PARAMS}}
    return DeFiContagionModel(shock_pcts=dict(shocks), **p)


def _snapshot(model: DeFiContagionModel) -> Dict[str, float]:
    """Phân loại người vay còn nợ theo HF cuối tick (người đã hết nợ coi là an toàn)."""
    bs = model.borrowers()
    live = [b for b in bs if b.debt > 0]
    hfs = [b.health_factor for b in live] or [999.0]
    return {
        "safe": sum(1 for b in bs if b.debt <= 0 or b.health_factor >= RISK_HF),
        "at_risk": sum(1 for b in live if 1.0 <= b.health_factor < RISK_HF),
        "liquidatable": sum(1 for b in live if b.health_factor < 1.0),
        "avg_hf": sum(hfs) / len(hfs), "min_hf": min(hfs),
        "total_debt": sum(b.debt for b in bs),
    }


def run_scenario(shocks: Dict[str, float], n_ticks: int = 15, **params) -> Tuple[DeFiContagionModel, pd.DataFrame]:
    """Chạy n_ticks và trả (mô hình, bảng theo tick gồm tick 0)."""
    model = build_model(shocks, **params)
    rows: List[dict] = []
    base = dict(tick=0, liquidated_borrowers=0, liquidation_events=0, liquidated_value_usd=0.0,
                debt_covered_usd=0.0, cum_liquidated_borrowers=0, bad_debt=0.0)
    rows.append({**base, **_snapshot(model), **{f"price_{a}": model.oracle_prices[a] for a in ASSETS}})
    seen = set()
    for t in range(1, n_ticks + 1):
        model.step()
        ev = [e for e in model.liquidation_events if e["tick"] == t]
        seen |= {e["borrower_id"] for e in ev}
        rows.append({
            "tick": t,
            "liquidated_borrowers": len({e["borrower_id"] for e in ev}),
            "liquidation_events": len(ev),
            # giá trị thế chấp bị tịch thu = nợ trả hộ x (1 + thưởng); ở đây dùng đúng công thức của mô hình
            "liquidated_value_usd": sum(e["debt_covered"] * (1.0 + model.liquidation_bonuses[e["seized_asset"]])
                                        for e in ev),
            "debt_covered_usd": sum(e["debt_covered"] for e in ev),
            "cum_liquidated_borrowers": len(seen),
            "bad_debt": model.bad_debt,
            **_snapshot(model),
            **{f"price_{a}": model.oracle_prices[a] for a in ASSETS},
        })
    return model, pd.DataFrame(rows).set_index("tick", drop=False).rename_axis(None)


def amplification(shocks: Dict[str, float], n_ticks: int = 15, **params) -> float:
    """
    Hệ số khuếch đại của kênh AMM: nợ thanh lý khi oracle bám AMM / khi oracle theo giá ngoại sinh
    (cùng seed, cùng cú sốc). Dùng được cho cả hai kịch bản; hàm indirect_share không dùng được cho sốc
    đa tài sản vì khi cả ba tài sản bị sốc thì nhóm nào cũng là nhóm "trực tiếp".
    """
    on, _ = run_scenario(shocks, n_ticks, **{**params, "oracle_mode": "amm"})
    off, _ = run_scenario(shocks, n_ticks, **{**params, "oracle_mode": "external"})
    d_on = sum(e["debt_covered"] for e in on.liquidation_events)
    d_off = sum(e["debt_covered"] for e in off.liquidation_events)
    return d_on / d_off if d_off > 0 else float("inf") if d_on > 0 else 1.0


def events_frame(model: DeFiContagionModel) -> pd.DataFrame:
    return pd.DataFrame(model.liquidation_events)
