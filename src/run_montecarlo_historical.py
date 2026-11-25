"""
run_montecarlo_historical.py - Monte Carlo với cú sốc lấy từ LỊCH SỬ GIÁ THẬT (historical calibration).

Quy trình
  1. Đọc prices_<TÀI SẢN>.csv (fetch_prices.py), tính phân phối cú sốc giảm giá h ngày của từng tài sản
     WETH, WBTC, LINK; mỗi lần chạy sốc đúng tài sản đó (shock_asset).
  2. Bốc ngẫu nhiên N cú sốc (có hoàn lại) từ các cửa sổ có giá GIẢM >= MIN_DROP; mỗi cú sốc chạy
     một mô phỏng với seed khác nhau (seed thay đổi cả quần thể người vay).
  3. Báo cáo: xác suất xảy ra lây lan, nợ xấu trung bình, VaR95 và ES95 của nợ xấu, bảng theo khoảng sốc.
  4. Chạy cho 2 cấu hình: gốc của nhóm và tham số Aave công bố (calibration.PUBLISHED_PARAMS).
  Kết quả CÓ ĐIỀU KIỆN: "nếu xảy ra cú sốc giảm >= MIN_DROP trong h ngày". Tần suất của điều kiện
  đó được in riêng từ dữ liệu, không nhân gộp vào VaR.

Chạy:  python run_montecarlo_historical.py [--dir .] [--assets WETH WBTC LINK] [--h 7] [--n 300]
"""
import argparse
import random

import os

from calibration import PUBLISHED_PARAMS, h_day_drops, load_dates, load_prices, var_es, worst_episodes
from model import DeFiContagionModel, GROUP_ASSETS
from run_experiment import indirect_share

N_TICKS = 15
MIN_DROP = 0.10
BINS = ((0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.99))
CONFIGS = {"Cấu hình gốc": {}, "Tham số Aave công bố": PUBLISHED_PARAMS}


def run_shock(shock: float, seed: int, overrides: dict, asset: str = "WETH") -> dict:
    model = DeFiContagionModel(seed=seed, shock_pct=shock, shock_asset=asset, **overrides)
    for _ in range(N_TICKS):
        model.step()
    ev = model.liquidation_events
    direct = {g for g, assets in GROUP_ASSETS.items() if asset in assets}   # nhóm có giữ tài sản bị sốc
    indirect_debt = sum(e["debt_covered"] for e in ev if e["group_id"] not in direct)
    return {"shock": shock, "bad_debt": model.bad_debt, "indirect_share": indirect_share(model),
            "contagion": indirect_debt > 0, "liquidations": len(ev)}


def sample_shocks(drops, n: int, rng: random.Random):
    tail = [d for d in drops if d >= MIN_DROP]
    if not tail:
        raise ValueError(f"Không có cửa sổ nào giảm >= {MIN_DROP:.0%} trong dữ liệu")
    return [min(0.95, rng.choice(tail)) for _ in range(n)]


def summarize(name: str, runs: list, asset: str) -> dict:
    losses = [r["bad_debt"] for r in runs]
    var, es = var_es(losses, 0.95)
    p_c = sum(r["contagion"] for r in runs) / len(runs)
    print(f"\n  [{name}]  {len(runs)} lần chạy")
    print(f"    Xác suất lây lan (nợ thanh lý ở nhóm không giữ {asset} > 0): {p_c:.1%}")
    print(f"    Nợ xấu: trung bình {sum(losses) / len(losses):,.0f} | VaR95 {var:,.0f} | ES95 {es:,.0f} USD")
    print(f"    {'Khoảng sốc':<12}{'Số lần':>8}{'P(lây lan)':>12}{'Nợ xấu TB':>14}")
    for lo, hi in BINS:
        sel = [r for r in runs if lo <= r["shock"] < hi]
        if sel:
            print(f"    {lo:.0%}-{min(hi, 1):.0%}{'':<5}{len(sel):>8}"
                  f"{sum(r['contagion'] for r in sel) / len(sel):>12.0%}"
                  f"{sum(r['bad_debt'] for r in sel) / len(sel):>14,.0f}")
    return {"p": p_c, "mean": sum(losses) / len(losses), "var": var, "es": es}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="thư mục chứa prices_WETH.csv, prices_WBTC.csv, prices_LINK.csv")
    ap.add_argument("--assets", nargs="+", default=["WETH", "WBTC", "LINK"])
    ap.add_argument("--h", type=int, default=7, help="độ dài cửa sổ giảm giá (ngày)")
    ap.add_argument("--n", type=int, default=300, help="số cú sốc bốc mẫu")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    summary = {}
    for asset in a.assets:
        path = os.path.join(a.dir, f"prices_{asset}.csv")
        prices, dates = load_prices(path), load_dates(path)
        drops = h_day_drops(prices, a.h)
        tail = [d for d in drops if d >= MIN_DROP]
        print(f"\n==================== {asset} ====================")
        print(f"Dữ liệu: {len(prices)} ngày ({dates[0]} đến {dates[-1]}), {len(drops)} cửa sổ {a.h} ngày")
        print(f"Tần suất cửa sổ giảm >= {MIN_DROP:.0%}: {len(tail)}/{len(drops)} = {len(tail) / len(drops):.1%}")
        print("Giai đoạn giảm mạnh nhất (không chồng lấn):")
        for d0, d1, dr in worst_episodes(dates, prices, a.h):
            print(f"  {d0} -> {d1}: giảm {dr:.1%}")
        rng = random.Random(a.seed)
        shocks = sample_shocks(drops, a.n, rng)
        for name, overrides in CONFIGS.items():
            runs = [run_shock(s, 1000 + i, overrides, asset) for i, s in enumerate(shocks)]
            summary[(asset, name)] = summarize(name, runs, asset)
            summary[(asset, name)]["freq"] = len(tail) / len(drops)
    print("\n==================== TỔNG HỢP (cấu hình gốc) ====================")
    print(f"{'Tài sản':<8}{'Tần suất sốc>=10%':>19}{'P(lây lan)':>12}{'Nợ xấu TB':>12}{'VaR95':>10}{'ES95':>10}")
    for asset in a.assets:
        r = summary[(asset, "Cấu hình gốc")]
        print(f"{asset:<8}{r['freq']:>19.1%}{r['p']:>12.1%}{r['mean']:>12,.0f}{r['var']:>10,.0f}{r['es']:>10,.0f}")
    print("\nLưu ý: cửa sổ trượt chồng lấn nên các cú sốc bốc mẫu không độc lập hoàn toàn;")
    print("mỗi lần chỉ sốc MỘT tài sản (các tài sản khác chỉ giảm qua kênh lây lan).")


if __name__ == "__main__":
    main()
