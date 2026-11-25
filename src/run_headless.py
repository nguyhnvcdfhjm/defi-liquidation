"""
run_headless.py - Script thực thi mô phỏng chế độ dòng lệnh (Headless Mode).
"""
import argparse

import pandas as pd

from model import DeFiContagionModel


def run_simulation(
    n_borrowers: int = 60,
    shock_asset: str = "WETH",
    shock_pct: float = 0.35,
    amm_depth: float = 1.0,
    n_ticks: int = 15,
    seed: int = 42,
    output_csv: str = "simulation_results.csv",
    close_factor: float = 0.50,
    verbose: bool = True,
) -> tuple:
    """Khởi tạo mô hình, chạy n_ticks, in tiến trình từng tick ra console và xuất file CSV.

    Returns: (model, DataFrame số liệu theo tick).
    """
    model = DeFiContagionModel(
        n_borrowers=n_borrowers, shock_asset=shock_asset, shock_pct=shock_pct,
        amm_depth_multiplier=amm_depth, close_factor=close_factor, seed=seed)
    for _ in range(n_ticks):
        model.step()
        if verbose:
            p = model.oracle_prices
            print(f"Tick {model.current_tick:>2} | WETH ${p['WETH']:>9,.0f} | WBTC ${p['WBTC']:>9,.0f} | "
                  f"LINK ${p['LINK']:>6,.2f} | thanh lý {model.total_liquidations:>4} | "
                  f"nợ xấu ${model.bad_debt:>11,.0f}")
    df = model.datacollector.get_model_vars_dataframe()
    if output_csv:
        df.to_csv(output_csv, index_label="tick")
        if verbose:
            print(f"Đã xuất: {output_csv}")
    return model, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=15)
    ap.add_argument("--borrowers", type=int, default=60)
    ap.add_argument("--shock-asset", default="WETH")
    ap.add_argument("--shock-pct", type=float, default=0.35)
    ap.add_argument("--depth", type=float, default=1.0)
    ap.add_argument("--close-factor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="simulation_results.csv")
    a = ap.parse_args()
    run_simulation(a.borrowers, a.shock_asset, a.shock_pct, a.depth, a.ticks, a.seed,
                   a.csv, a.close_factor)


if __name__ == "__main__":
    main()
