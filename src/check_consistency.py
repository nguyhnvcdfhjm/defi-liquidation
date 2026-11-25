"""
check_consistency.py - Kiểm tra số liệu trên giao diện, script và dữ liệu gốc có trùng khớp không.

Chạy trong thư mục dự án:   python check_consistency.py
Mỗi mục in KHỚP hoặc LỆCH. Thoát với mã 1 nếu có mục LỆCH.

Các nhóm kiểm tra:
  A. Một lần chạy (seed 42): thẻ chỉ số, nhật ký thanh lý, DataCollector và file CSV phải cùng một số.
  B. Bất biến của mô hình: tích k của bể không giảm, đấu giá có phí >= 0, nợ xấu >= 0.
  C. Giao diện so với script: bảng Độ nhạy và Cạnh tranh gas trong app.py so với run_sensitivity.py
     và run_competition.py (cần cài streamlit và plotly; thiếu thì bỏ qua nhóm này).
  D. Số tham chiếu đã xác nhận trên Mesa thật (seed 42, cấu hình mặc định).
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from model import DeFiContagionModel
from run_experiment import indirect_share, mean_ci95
import run_competition
import run_headless
import run_sensitivity

TOL = 1e-6            # sai số tuyệt đối cho so sánh số thực
RESULTS = []          # (nhóm, tên, đạt?, chi tiết)


def check(group: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((group, name, bool(ok), detail))
    print(f"  [{'KHỚP' if ok else 'LỆCH'}] {name}" + (f"  ({detail})" if detail else ""))


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=tol)


# --------------------------------------------------------------------------- A. một lần chạy
def check_single_run() -> None:
    print("\nA. Một lần chạy (seed 42, 15 tick): các nguồn số liệu phải khớp nhau")
    n_ticks = 15
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sim.csv")
        model, df = run_headless.run_simulation(seed=42, n_ticks=n_ticks, output_csv=csv_path, verbose=False)
        df_csv = pd.read_csv(csv_path, index_col="tick")

    events = model.liquidation_events
    last = df.iloc[-1]
    g = "A"
    check(g, "Số dòng DataCollector = số tick + 1 (gồm tick 0)", len(df) == n_ticks + 1, f"{len(df)} dòng")
    check(g, "Thẻ 'Tổng vụ thanh lý' = số dòng nhật ký thanh lý",
          model.total_liquidations == len(events), f"{model.total_liquidations} và {len(events)}")
    check(g, "Cột Total_Liquidations (tick cuối) = số dòng nhật ký",
          int(last["Total_Liquidations"]) == len(events), f"{int(last['Total_Liquidations'])}")
    check(g, "Thẻ 'Tổng nợ xấu' = cột Bad_Debt (tick cuối)",
          close(model.bad_debt, float(last["Bad_Debt"])), f"{model.bad_debt:,.2f}")
    for asset in ("WETH", "WBTC", "LINK"):
        check(g, f"Giá {asset} trên biểu đồ (tick cuối) = oracle trong mô hình",
              close(float(last[f"Price_{asset}"]), model.oracle_prices[asset]), f"{model.oracle_prices[asset]:,.4f}")
    check(g, "File CSV = DataCollector (mọi ô số)",
          df_csv.shape == df.shape and bool(((df_csv - df.astype(float)).abs().fillna(0) < 1e-6).all().all()),
          f"{df_csv.shape[0]} dòng x {df_csv.shape[1]} cột")
    check(g, "Mọi sự kiện thanh lý có tick trong [1, số tick]",
          all(1 <= e["tick"] <= n_ticks for e in events))
    check(g, "Số lần thanh lý theo từng tick cộng lại = tổng",
          sum(sum(1 for e in events if e["tick"] == t) for t in range(1, n_ticks + 1)) == len(events))
    share_sum = sum(e["debt_covered"] for e in events if e["group_id"] != 0) / sum(e["debt_covered"] for e in events)
    check(g, "Tỷ trọng gián tiếp (hàm) = tính lại từ nhật ký",
          close(indirect_share(model), share_sum), f"{100 * share_sum:.2f}%")


# --------------------------------------------------------------------------- B. bất biến
def check_invariants() -> None:
    print("\nB. Bất biến của mô hình")
    g = "B"
    model = DeFiContagionModel(seed=42)
    k0 = {a: p.k for a, p in model.amm_pools.items()}
    k_ok = True
    for _ in range(15):
        model.step()
        for a, p in model.amm_pools.items():
            # Arbitrageur giữ k; swap có phí làm k tăng nhẹ; k không được giảm đáng kể
            if p.k < k0[a] * (1 - 1e-9):
                k_ok = False
    check(g, "Tích k của mọi bể không giảm sau 15 tick", k_ok)
    check(g, "Nợ xấu >= 0", model.bad_debt >= 0, f"{model.bad_debt:,.2f}")
    check(g, "Mọi dự trữ bể > 0", all(p.reserve_token > 0 and p.reserve_usdc > 0 for p in model.amm_pools.values()))

    m3 = DeFiContagionModel(seed=42, n_liquidators=5)
    for _ in range(15):
        m3.step()
    ev = m3.liquidation_events
    check(g, "5 bot: phí ưu tiên >= 0 và n_bidders = 5 ở mọi sự kiện",
          all(e["priority_fee"] >= 0 and e["n_bidders"] == 5 for e in ev), f"{len(ev)} sự kiện")

    a = DeFiContagionModel(seed=7)
    b = DeFiContagionModel(seed=7)
    for _ in range(15):
        a.step()
        b.step()
    check(g, "Cùng seed cho cùng kết quả (tái lập)",
          a.total_liquidations == b.total_liquidations and close(a.bad_debt, b.bad_debt))


# --------------------------------------------------------------------------- C. giao diện so với script
def check_ui_vs_scripts() -> None:
    print("\nC. Giao diện (app.py) so với script dòng lệnh")
    g = "C"
    try:
        import app  # cần streamlit và plotly
    except Exception as exc:  # noqa: BLE001 - thiếu thư viện giao diện thì bỏ qua nhóm này
        print(f"  [BỎ QUA] không nạp được app.py ({type(exc).__name__}: {exc})")
        return

    check(g, "Số seed giao diện = số seed run_sensitivity.py",
          app.N_SEEDS_UI == run_sensitivity.N_SEEDS, f"{app.N_SEEDS_UI} và {run_sensitivity.N_SEEDS}")
    check(g, "Số seed giao diện = số seed run_competition.py",
          app.N_SEEDS_UI == run_competition.N_SEEDS, f"{app.N_SEEDS_UI} và {run_competition.N_SEEDS}")
    check(g, "Số tick giao diện = số tick các script",
          app.N_TICKS_UI == run_sensitivity.N_TICKS == run_competition.N_TICKS)

    # --- Độ nhạy
    ui = app.compute_sensitivity.__wrapped__(app.N_SEEDS_UI) if hasattr(app.compute_sensitivity, "__wrapped__") \
        else app.compute_sensitivity(app.N_SEEDS_UI)
    all_ok = True
    for param, (label, values) in run_sensitivity.SWEEPS.items():
        for v in values:
            runs = [run_sensitivity.run_one(s, **{param: v}) for s in range(run_sensitivity.N_SEEDS)]
            share, share_ci = mean_ci95([r["indirect_share"] * 100 for r in runs])
            bad, bad_ci = mean_ci95([r["bad_debt"] / 1e3 for r in runs])
            row = ui[(ui["Tham số"] == label) & (ui["Giá trị"] == v)].iloc[0]
            ok = (close(row["Tỷ trọng gián tiếp (%)"], share) and close(row["CI95 tỷ trọng"], share_ci)
                  and close(row["Nợ xấu (nghìn $)"], bad) and close(row["CI95 nợ xấu"], bad_ci))
            all_ok &= ok
            check(g, f"Độ nhạy {label} = {v}", ok,
                  f"tỷ trọng {share:.1f} ± {share_ci:.1f}; nợ xấu {bad:.1f} ± {bad_ci:.1f}")
    check(g, "Bảng Độ nhạy: mọi dòng khớp script", all_ok)

    # --- Cạnh tranh gas
    comp = app.compute_competition.__wrapped__(app.N_SEEDS_UI) if hasattr(app.compute_competition, "__wrapped__") \
        else app.compute_competition(app.N_SEEDS_UI)
    all_ok = True
    for n in run_competition.N_LIST:
        runs = [run_competition.run_one(n, s) for s in range(run_competition.N_SEEDS)]
        events = sum(r["n_events"] for r in runs)
        profit_sum = sum(r["profit_sum"] for r in runs)
        fee = sum(r["fee_sum"] for r in runs)
        profit = profit_sum / events
        fee_share = 100 * fee / (fee + profit_sum)
        share = mean_ci95([r["indirect_share"] * 100 for r in runs])[0]
        row = comp[comp["Số bot"] == n].iloc[0]
        ok = (close(row["Lợi nhuận TB/khoản (USD)"], profit) and close(row["Phí ưu tiên / thặng dư (%)"], fee_share)
              and close(row["Tỷ trọng gián tiếp (%)"], share))
        all_ok &= ok
        check(g, f"Cạnh tranh gas n = {n}", ok,
              f"lợi nhuận {profit:.2f}; phí/thặng dư {fee_share:.1f}%; tỷ trọng {share:.1f}%")
    check(g, "Bảng Cạnh tranh gas: mọi dòng khớp script", all_ok)

    # --- Stackelberg
    import run_stackelberg
    ui_grid = app.compute_stackelberg.__wrapped__(app.N_SEEDS_UI) if hasattr(app.compute_stackelberg, "__wrapped__") \
        else app.compute_stackelberg(app.N_SEEDS_UI)
    ref = run_stackelberg.build_grid(run_stackelberg.N_SEEDS)
    check(g, "Số seed giao diện = số seed run_stackelberg.py", app.N_SEEDS_UI == run_stackelberg.N_SEEDS)
    same = all(close(float(ui_grid[c].iloc[i]), float(ref[c].iloc[i]))
               for c in ("kappa", "bonus_scale", "bad_debt", "bonus_paid", "unresolved") for i in range(len(ref)))
    check(g, "Lưới Stackelberg: mọi ô khớp script", same and len(ui_grid) == len(ref))


# --------------------------------------------------------------------------- D. số tham chiếu
def check_reference() -> None:
    print("\nD. Số tham chiếu (seed 42, cấu hình mặc định, 15 tick; đã xác nhận trên Mesa thật)")
    g = "D"
    model = DeFiContagionModel(seed=42)
    for _ in range(15):
        model.step()
    check(g, "Tổng vụ thanh lý = 185", model.total_liquidations == 185, f"{model.total_liquidations}")
    check(g, "Tổng nợ xấu = 291.012 USD (làm tròn)", round(model.bad_debt) == 291012, f"{model.bad_debt:,.0f}")
    weth = 100 * (model.oracle_prices["WETH"] / model.baseline_prices["WETH"] - 1)
    wbtc = 100 * (model.oracle_prices["WBTC"] / model.baseline_prices["WBTC"] - 1)
    check(g, "WETH thay đổi -44,2% so với giá gốc", round(weth, 1) == -44.2, f"{weth:.1f}%")
    check(g, "WBTC thay đổi -30,5% so với giá gốc", round(wbtc, 1) == -30.5, f"{wbtc:.1f}%")


def main() -> int:
    print("Kiểm tra tính nhất quán số liệu")
    check_single_run()
    check_invariants()
    check_ui_vs_scripts()
    check_reference()
    bad = [r for r in RESULTS if not r[2]]
    print(f"\nTổng: {len(RESULTS) - len(bad)} khớp, {len(bad)} lệch")
    if bad:
        print("Các mục LỆCH:")
        for grp, name, _, detail in bad:
            print(f"  - [{grp}] {name} {detail}")
        print("Gợi ý: thay app.py bản mới, xóa cache Streamlit (phím C trên trang), chạy lại.")
        return 1
    print("Mọi mục đều khớp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
