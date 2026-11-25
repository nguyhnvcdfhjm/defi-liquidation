"""
run_stackelberg.py - Trò chơi Stackelberg: GIAO THỨC (leader) chọn tham số, THANH LÝ VIÊN (follower) phản ứng.

Cấu trúc trò chơi
  Leader   : giao thức cho vay. Chọn trước (kappa, s):
               kappa = close factor (tỷ lệ nợ tối đa được thanh lý mỗi lần),
               s     = hệ số nhân của thưởng thanh lý, thưởng_k = s * thưởng_gốc_k.
  Follower : thanh lý viên (cùng người vay và bot arbitrage, đều theo quy tắc cố định). Thanh lý viên quan sát
             (kappa, s) rồi hành động theo quy tắc tham lam trong agents.py: chọn tài sản k* có Pi cao nhất,
             chỉ thanh lý khi Pi = gộp - gas > 0. Quy tắc này là tối ưu TỪNG KHOẢN, KHÔNG phải phản ứng tối ưu
             chiến lược: nó không dự đoán tác động giá lên các lần thanh lý sau và không chọn việc chờ.
  Bài toán của leader:   min  L(kappa, s) = NợXấu + w * ThưởngĐãTrả
                         s.t.  TồnĐọng(kappa, s) <= RHO              (RHO thử 10-25%)
                               ThamGia(kappa, s) >= PART_MIN         (PART_MIN = 50% khối lượng nợ thanh lý của ô gốc)
             - NợXấu       : rủi ro mất khả năng thanh toán của giao thức (chỉ đếm khi có thanh lý).
             - ThưởngĐãTrả : phần giá trị người vay bị trừ cho thanh lý viên (chi phí của người dùng).
             - w           : mức coi trọng chi phí người dùng so với solvency (leader chọn).
             - TồnĐọng     : nợ của người vay có HF < 1 vẫn chưa bị thanh lý sau N tick, tính theo % tổng nợ ban đầu.
             - ThamGia     : khối lượng nợ đã thanh lý của ô chia khối lượng của ô gốc (kappa=0.5, s=1.0). Nếu thưởng quá thấp,
                             thanh lý viên có Pi <= 0 nên không tham gia; NợXấu ghi nhận bằng 0 nhưng rủi ro còn
                             nguyên. Ngưỡng TồnĐọng đơn lẻ không loại được các ô này (tồn đọng ~14% < ô gốc 18.7%),
                             nên cần ràng buộc tham gia riêng. PART_MIN là GIẢ ĐỊNH chưa kiểm chứng; bảng cuối
                             in nghiệm theo nhiều mức PART_MIN để thấy độ nhạy.

Cách giải (tìm kiếm trên lưới bằng mô phỏng, KHÔNG phải quy nạp ngược chặt chẽ)
  Với mỗi cặp (kappa, s) trên lưới 5x5, chạy N_SEEDS seed (cùng dãy seed cho mọi ô = common random numbers).
  Trung bình L ước lượng L(kappa, s) khi follower chạy theo quy tắc cố định. Leader chọn ô L nhỏ nhất trong các ô thỏa
  ràng buộc. Vì follower là heuristic nên đây là "tối ưu leader trên mô phỏng", không phải cân bằng Stackelberg
  chặt chẽ; kết quả phụ thuộc mô hình, không phải khuyến nghị cho Aave thật. Cột bad_debt_vs_base_ci là khoảng tin cậy
  GHÉP CẶP theo seed so với ô gốc; bad_debt_ci là khoảng tin cậy không ghép cặp của riêng từng ô.
  Chưa có tách seed train/test: chọn nghiệm và báo cáo dùng cùng 10 seed.

Chạy:  python run_stackelberg.py            (mặc định 10 seed, 15 tick, 25 ô lưới)
"""
import argparse
import itertools

import pandas as pd

from model import DeFiContagionModel
from run_experiment import indirect_share, mean_ci95

N_SEEDS = 10
N_TICKS = 15
KAPPAS = (0.10, 0.25, 0.50, 0.75, 1.00)
BONUS_SCALES = (0.25, 0.5, 1.0, 1.5, 2.0)
RHO = 0.15                        # tồn đọng tối đa mặc định (15% tổng nợ ban đầu); ký hiệu rho, khác epsilon của Nash. ρ = 10% không có ô nào thỏa khi PART = 0.5
WEIGHTS = (0.0, 1.0)             # w trong hàm mất mát
RHO_LIST = (0.10, 0.15, 0.20, 0.25)
BASE_CELL = (0.50, 1.0)          # (kappa, s) gốc, mốc cho mức tham gia và so sánh ghép cặp
PART_MIN = 0.5                   # tham gia tối thiểu: khối lượng nợ thanh lý >= 50% của ô gốc (GIẢ ĐỊNH)
PART_LIST = (0.0, 0.25, 0.5, 0.75)


def run_one(seed: int, kappa: float, scale: float, n_liquidators: int = 1) -> dict:
    """Một lần chơi: leader đã chọn (kappa, scale); follower phản ứng trong mô hình."""
    base = DeFiContagionModel(seed=seed, n_liquidators=n_liquidators).liquidation_bonuses
    bonuses = {a: b * scale for a, b in base.items()}
    model = DeFiContagionModel(seed=seed, close_factor=kappa, liquidation_bonuses=bonuses,
                               n_liquidators=n_liquidators)
    debt0 = sum(b.debt for b in model.borrowers())
    for _ in range(N_TICKS):
        model.step()
    unresolved = sum(b.debt for b in model.borrowers() if b.debt > 0 and b.health_factor < 1.0) / debt0
    ev = model.liquidation_events
    bonus_paid = sum(e["debt_covered"] * bonuses[e["seized_asset"]] for e in ev)
    liq_debt = sum(e["debt_covered"] for e in ev) / debt0
    return {"bad_debt": model.bad_debt, "bonus_paid": bonus_paid, "liquidations": len(ev), "liq_debt": liq_debt,
            "stalled": model.stalled_liquidations, "unresolved": unresolved, "indirect_share": indirect_share(model)}


def build_grid(n_seeds: int = N_SEEDS, n_liquidators: int = 1) -> pd.DataFrame:
    cells = {(k, sc): [run_one(s, k, sc, n_liquidators) for s in range(n_seeds)]
             for k, sc in itertools.product(KAPPAS, BONUS_SCALES)}
    base_runs = cells[BASE_CELL]
    base_liq = sum(r["liq_debt"] for r in base_runs) / n_seeds
    rows = []
    for (kappa, scale), runs in cells.items():
        row = {"kappa": kappa, "bonus_scale": scale}
        for key in ("bad_debt", "bonus_paid", "liquidations", "liq_debt", "stalled", "unresolved", "indirect_share"):
            row[key] = sum(r[key] for r in runs) / n_seeds
        row["bad_debt_ci"] = mean_ci95([r["bad_debt"] for r in runs])[1]
        diffs = [r["bad_debt"] - b["bad_debt"] for r, b in zip(runs, base_runs)]
        row["bad_debt_vs_base"] = sum(diffs) / n_seeds
        row["bad_debt_vs_base_ci"] = mean_ci95(diffs)[1] if any(d != 0 for d in diffs) else 0.0
        row["participation"] = row["liq_debt"] / base_liq if base_liq else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def feasible_cells(grid: pd.DataFrame, rho: float, part_min: float) -> pd.DataFrame:
    return grid[(grid["unresolved"] <= rho) & (grid["participation"] >= part_min)]


def best_response_table(grid: pd.DataFrame, weights=WEIGHTS, rho_list=RHO_LIST, part_list=(PART_MIN,)) -> pd.DataFrame:
    """Với mỗi (RHO, PART_MIN, w), leader chọn (kappa, s) làm L nhỏ nhất trong các ô thỏa cả hai ràng buộc."""
    out = []
    for rho in rho_list:
        for part in part_list:
            feasible = feasible_cells(grid, rho, part)
            for w in weights:
                if feasible.empty:
                    out.append({"RHO": rho, "PART": part, "w": w, "kappa*": None, "bonus_scale*": None})
                    continue
                loss = feasible["bad_debt"] + w * feasible["bonus_paid"]
                i = loss.idxmin()
                r = grid.loc[i]
                out.append({"RHO": rho, "PART": part, "w": w, "kappa*": r["kappa"], "bonus_scale*": r["bonus_scale"],
                            "L*": loss[i], "bad_debt": r["bad_debt"], "bonus_paid": r["bonus_paid"],
                            "tồn đọng": r["unresolved"], "tham gia": r["participation"]})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-liquidators", type=int, default=1)
    ap.add_argument("--csv", default="stackelberg_grid.csv")
    a = ap.parse_args()

    grid = build_grid(a.seeds, a.n_liquidators)
    grid.to_csv(a.csv, index=False, encoding="utf-8-sig")
    print(f"{a.seeds} seed, {N_TICKS} tick, {a.n_liquidators} thanh lý viên; ô mặc định "
          f"(kappa=0.5, s=1.0) là tham số gốc\n")
    print("Nợ xấu trung bình (USD): hàng = kappa, cột = hệ số thưởng s")
    print(grid.pivot(index="kappa", columns="bonus_scale", values="bad_debt").round(0).to_string())
    print("\nThưởng đã trả trung bình (USD)")
    print(grid.pivot(index="kappa", columns="bonus_scale", values="bonus_paid").round(0).to_string())
    print("\nNợ tồn đọng cuối kỳ (HF<1 chưa bị thanh lý, % nợ ban đầu); ô > "
          f"{RHO:.0%} bị loại")
    print((100 * grid.pivot(index="kappa", columns="bonus_scale", values="unresolved")).round(1).to_string())
    print("\nMức tham gia của thanh lý viên (khối lượng nợ thanh lý / ô gốc, %)")
    print((100 * grid.pivot(index="kappa", columns="bonus_scale", values="participation")).round(0).to_string())
    print("\nChênh lệch nợ xấu so với ô gốc, ghép cặp theo seed (USD, nửa khoảng tin cậy 95% ở bảng sau)")
    print(grid.pivot(index="kappa", columns="bonus_scale", values="bad_debt_vs_base").round(0).to_string())
    print(grid.pivot(index="kappa", columns="bonus_scale", values="bad_debt_vs_base_ci").round(0).to_string())
    print(f"\nLựa chọn tối ưu của leader theo w, ngưỡng tồn đọng RHO và mức tham gia tối thiểu PART "
          f"(mặc định PART = {PART_MIN}; L = nợ xấu + w * thưởng)")
    print(best_response_table(grid, part_list=PART_LIST).round(
        {"kappa*": 2, "bonus_scale*": 2, "L*": 0, "bad_debt": 0, "bonus_paid": 0, "tồn đọng": 3, "tham gia": 2}
    ).to_string(index=False))
    base = grid[(grid.kappa == 0.5) & (grid.bonus_scale == 1.0)].iloc[0]
    print(f"\nĐã lưu {a.csv}. Tham số gốc: nợ xấu {base.bad_debt:,.0f}, thưởng {base.bonus_paid:,.0f}")


if __name__ == "__main__":
    main()
