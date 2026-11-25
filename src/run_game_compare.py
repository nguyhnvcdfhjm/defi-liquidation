"""
run_game_compare.py - So sánh TRƯỚC và SAU khi tích hợp module lý thuyết trò chơi (game_theory.py).

Thiết kế (so sánh cặp, cùng seed => cùng quần thể người vay ban đầu, cùng cú sốc):
    baseline       : mô hình gốc (người vay "active" trả nợ theo luật cố định 1 < HF < 1,15)
    nash_active    : đúng nhóm active (trung bình khoảng 18/60 người, 24 ở seed 42) nay là người chơi tối ưu (cân bằng Nash)
    planner_active : cùng nhóm, nhưng một nhà hoạch định chọn chung tỷ lệ trả nợ (mốc hợp tác)
    nash_all       : mọi người vay đều là người chơi
    planner_all    : mọi người vay, nhà hoạch định

Giá của sự vô chính phủ (PoA) = thiệt hại xã hội(nash) / thiệt hại xã hội(planner), cùng tập người chơi.
Thiệt hại xã hội = thưởng thanh lý người vay mất + nợ xấu + phi * tổng tiền trả trước (phi mặc định 0,03,
dùng cho CẢ baseline để cùng thước đo).

Chạy:
    python run_game_compare.py                 # 30 seed, 15 tick, sốc WETH -35%, kèm quét độ nhạy
    python run_game_compare.py --seeds 10 --skip-sensitivity
Kết quả: game_results.csv, game_summary.csv, game_sensitivity.csv, game_compare.png (nếu có matplotlib).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
from typing import Dict, List, Optional

import pandas as pd

from game_theory import GameConfig, GameTheoryModel
from run_experiment import indirect_share

PHI_METRIC = 0.03   # phi dùng để quy đổi chi phí trả nợ trong thước đo thiệt hại xã hội

ARMS = {
    "baseline": None,
    "nash_active": GameConfig(mode="nash", players="active"),
    "planner_active": GameConfig(mode="planner", players="active"),
    "nash_all": GameConfig(mode="nash", players="all"),
    "planner_all": GameConfig(mode="planner", players="all"),
}


def run_one(seed: int, game: Optional[GameConfig], n_ticks: int = 15, shock_pct: float = 0.35,
            **model_kw) -> Dict[str, float]:
    m = GameTheoryModel(seed=seed, shock_pct=shock_pct, game=game, **model_kw)
    cash0 = {b.unique_id: b.cash_usdc for b in m.borrowers()}
    for _ in range(n_ticks):
        m.step()
    ev = m.liquidation_events
    repaid = sum(cash0[b.unique_id] - b.cash_usdc for b in m.borrowers())
    bonus_paid = sum(e["debt_covered"] * m.liquidation_bonuses[e["seized_asset"]] for e in ev)
    df = m.datacollector.get_model_vars_dataframe()
    log = m.game.log if m.game else []
    return {
        "liq_events": len(ev),
        "liq_borrowers": len({e["borrower_id"] for e in ev}),
        "liquidated_debt": sum(e["debt_covered"] for e in ev),
        "bad_debt": m.bad_debt,
        "indirect_share": indirect_share(m),
        "bonus_paid": bonus_paid,
        "repaid": repaid,
        "social_cost": bonus_paid + m.bad_debt + PHI_METRIC * repaid,
        "min_WBTC": float(df["Price_WBTC"].min()),
        "min_LINK": float(df["Price_LINK"].min()),
        "last_liq_tick": max((e["tick"] for e in ev), default=0),
        "eps_max": max((l["epsilon"] for l in log), default=0.0),
        "not_converged": sum(not l["converged"] for l in log),
    }


def mean_ci(values: List[float]):
    m = statistics.mean(values)
    ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, ci


METRICS = [
    ("liquidated_debt", "Nợ đã thanh lý (nghìn $)", 1e-3),
    ("bad_debt", "Nợ xấu (nghìn $)", 1e-3),
    ("liq_borrowers", "Số người vay bị thanh lý", 1.0),
    ("indirect_share", "Tỷ trọng thanh lý gián tiếp (%)", 100.0),
    ("bonus_paid", "Thưởng thanh lý người vay mất (nghìn $)", 1e-3),
    ("repaid", "Tiền trả nợ trước (nghìn $)", 1e-3),
    ("social_cost", "Thiệt hại xã hội (nghìn $)", 1e-3),
    ("min_WBTC", "Giá WBTC thấp nhất ($)", 1.0),
    ("min_LINK", "Giá LINK thấp nhất ($)", 1.0),
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng tóm tắt: trung bình ± CI95 theo nhánh, và chênh lệch cặp so với baseline."""
    rows = []
    base = df[df.arm == "baseline"].set_index("seed")
    for arm in ARMS:
        d = df[df.arm == arm].set_index("seed")
        row = {"arm": arm}
        for key, _label, scale in METRICS:
            m, ci = mean_ci(list(d[key] * scale))
            row[key] = f"{m:,.1f} ± {ci:,.1f}"
        if arm != "baseline":
            diff = (d["bad_debt"] - base["bad_debt"]) * 1e-3
            m, ci = mean_ci(list(diff))
            row["Δ nợ xấu vs gốc (nghìn $)"] = f"{m:+,.1f} ± {ci:,.1f}"
            row["% seed nợ xấu giảm"] = f"{100 * (diff < 0).mean():.0f}"
            sc = (d["social_cost"] - base["social_cost"]) * 1e-3
            m, ci = mean_ci(list(sc))
            row["Δ thiệt hại xã hội vs gốc (nghìn $)"] = f"{m:+,.1f} ± {ci:,.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def poa_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for grp in ("active", "all"):
        n = df[df.arm == f"nash_{grp}"].set_index("seed")["social_cost"]
        p = df[df.arm == f"planner_{grp}"].set_index("seed")["social_cost"]
        ratio = (n / p).replace([math.inf], math.nan).dropna()
        out.append({"nhóm người chơi": grp,
                    "thiệt hại Nash (nghìn $)": n.mean() * 1e-3,
                    "thiệt hại planner (nghìn $)": p.mean() * 1e-3,
                    "PoA (tỷ số trung bình, mỗi seed)": ratio.mean(),
                    "PoA (tỷ số của trung bình)": n.mean() / p.mean()})
    return pd.DataFrame(out)


def sensitivity(seeds: int, n_ticks: int, shock_pct: float) -> pd.DataFrame:
    rows = []
    base = [run_one(s, None, n_ticks, shock_pct) for s in range(seeds)]
    b_bad = statistics.mean(r["bad_debt"] for r in base)
    b_soc = statistics.mean(r["social_cost"] for r in base)
    for grp in ("active", "all"):
        for param, values in (("phi", [0.0, 0.01, 0.03, 0.06, 0.12, 0.25]),
                              ("tau", [0.01, 0.03, 0.06, 0.12]),
                              ("horizon", [1, 2, 4, 6])):
            for v in values:
                cfg = GameConfig(mode="nash", players=grp, **{param: v})
                res = [run_one(s, cfg, n_ticks, shock_pct) for s in range(seeds)]
                rows.append({
                    "players": grp, "param": param, "value": v,
                    "bad_debt_k": statistics.mean(r["bad_debt"] for r in res) * 1e-3,
                    "bad_debt_vs_base_%": 100 * (statistics.mean(r["bad_debt"] for r in res) / b_bad - 1),
                    "social_cost_vs_base_%": 100 * (statistics.mean(r["social_cost"] for r in res) / b_soc - 1),
                    "repaid_k": statistics.mean(r["repaid"] for r in res) * 1e-3,
                    "eps_max": max(r["eps_max"] for r in res),
                })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str, shock_pct: float) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = list(ARMS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title, scale in ((axes[0], "bad_debt", "Nợ xấu (nghìn $)", 1e-3),
                                  (axes[1], "liquidated_debt", "Nợ đã thanh lý (nghìn $)", 1e-3),
                                  (axes[2], "social_cost", "Thiệt hại xã hội (nghìn $)", 1e-3)):
        means, cis = zip(*(mean_ci(list(df[df.arm == a][key] * scale)) for a in labels))
        colors = ["#888888", "#2a7ab8", "#7ab8e0", "#c2571a", "#e8a06b"]
        ax.bar(range(len(labels)), means, yerr=cis, color=colors, capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Trước (baseline) và sau khi thêm trò chơi giảm đòn bẩy  |  sốc WETH -{shock_pct:.0%}, "
                 f"{df.seed.nunique()} seed, cột lỗi = CI95", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--ticks", type=int, default=15)
    ap.add_argument("--shock-pct", type=float, default=0.35)
    ap.add_argument("--sens-seeds", type=int, default=10)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    rows = []
    for arm, cfg in ARMS.items():
        for s in range(a.seeds):
            rows.append({"arm": arm, "seed": s, **run_one(s, cfg, a.ticks, a.shock_pct)})
    df = pd.DataFrame(rows)
    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(os.path.join(a.outdir, "game_results.csv"), index=False)

    summ = summarize(df)
    summ.to_csv(os.path.join(a.outdir, "game_summary.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"{a.seeds} seed, {a.ticks} tick, sốc WETH -{a.shock_pct:.0%}\n")
    print(summ.set_index("arm").T.to_string())
    print("\nGiá của sự vô chính phủ:")
    print(poa_table(df).round(3).to_string(index=False))
    print(f"\nChất lượng nghiệm: epsilon lớn nhất = {df.eps_max.max():.4f} (tỷ lệ nợ), "
          f"số lần fictitious play không hội tụ = {int(df.not_converged.sum())}")

    if plot(df, os.path.join(a.outdir, "game_compare.png"), a.shock_pct):
        print("Đã vẽ game_compare.png")

    if not a.skip_sensitivity:
        sens = sensitivity(a.sens_seeds, a.ticks, a.shock_pct)
        sens.to_csv(os.path.join(a.outdir, "game_sensitivity.csv"), index=False)
        print("\nĐộ nhạy (Nash, %d seed):" % a.sens_seeds)
        print(sens.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
