"""
export_borrowers_csv.py - Xuất dữ liệu 60 người vay ra CSV và đối chiếu với thống kê của mô hình.

Chạy trong thư mục dự án (cạnh model.py):
    python export_borrowers_csv.py                     # seed 42, WETH -35%, 15 tick (cấu hình mặc định)
    python export_borrowers_csv.py --seed 7 --ticks 10
    python export_borrowers_csv.py --only-initial      # chỉ xuất trạng thái khởi tạo (tick 0)

Cách đọc cột HF: hf_before_liquidation là HF ngay trước vòng thanh lý của tick đó; health_factor là HF cuối tick.
Người vừa bị thanh lý có hf_before_liquidation < 1 nhưng health_factor >= 1 vì nợ đã giảm một nửa.

Tạo hai file (UTF-8 có BOM, mở bằng Excel không lỗi tiếng Việt):
    borrowers_tick0.csv      60 dòng: dữ liệu khởi tạo của từng người vay (đây là "dữ liệu đầu vào" của mô hình)
    borrowers_by_tick.csv    (tick + 1) x 60 dòng: trạng thái của từng người vay sau mỗi tick

Dữ liệu 60 người vay KHÔNG phải dữ liệu thật. Chúng được sinh ngẫu nhiên có kiểm soát trong
DeFiContagionModel._seed_borrowers theo seed, rồi hiệu chỉnh khoảng giá trị để cú sốc WETH -35% tạo thác giá.
Cuối chương trình có phần đối chiếu: số liệu trong CSV phải khớp thống kê của mô hình (tổng vụ thanh lý, nợ xấu,
số người mỗi nhóm, khoảng giá trị, HF ban đầu).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (DeFiContagionModel, GROUP_ASSETS, MAX_COLLATERAL_USD, MIN_COLLATERAL_USD,
                   SEED_HF_RANGE)

GROUP_NAMES = {0: "Nhóm 1 (WETH+WBTC)", 1: "Nhóm 2 (WBTC+LINK)", 2: "Nhóm 3 (LINK)"}
COLUMNS = [
    "tick", "borrower_id", "group_id", "group_name", "risk_profile",
    "weth_amount", "wbtc_amount", "link_amount",
    "collateral_value_usd", "debt_usd", "cash_usdc",
    "hf_before_liquidation", "health_factor", "status",
    "liquidated_this_tick", "debt_covered_this_tick_usd", "seized_this_tick",
    "n_liquidated_times", "bad_debt_recorded_usd",
]


def snapshot(model, tick, hf_before=None):
    """
    Trạng thái 60 người vay tại thời điểm hiện tại (tick 0 = trước khi chạy, tick >= 1 = sau model.step()).

    hf_before = {borrower_id: HF} chụp NGAY TRƯỚC vòng thanh lý của tick đó (sau cú sốc và sau bước Borrower).
    Cột health_factor là HF CUỐI tick (sau thanh lý và sau Arbitrageur), nên người vừa bị thanh lý có
    hf_before_liquidation < 1 nhưng health_factor >= 1 (nợ đã giảm một nửa).
    """
    n_liq = {}
    for e in model.liquidation_events:
        if e["tick"] <= tick:
            n_liq[e["borrower_id"]] = n_liq.get(e["borrower_id"], 0) + 1
    this_tick = {}
    for e in model.liquidation_events:
        if e["tick"] == tick:
            this_tick.setdefault(e["borrower_id"], []).append(e)
    rows = []
    for b in model.borrowers():
        hf = b.calculate_health_factor()
        hf0 = hf if hf_before is None else hf_before[b.unique_id]
        evs = this_tick.get(b.unique_id, [])
        if b.debt <= 0:
            status = "hết nợ"
        elif evs:
            status = "bị thanh lý trong tick này" + ("" if hf >= 1.0 else ", vẫn HF<1")
        elif hf < 1.0:
            status = "HF<1, chưa được thanh lý"
        elif b.is_liquidated:
            status = "đã bị thanh lý ở tick trước, HF>=1"
        else:
            status = "an toàn"
        rows.append({
            "tick": tick,
            "borrower_id": b.unique_id,
            "group_id": b.group_id,
            "group_name": GROUP_NAMES[b.group_id],
            "risk_profile": b.risk_profile,
            "weth_amount": round(b.collaterals.get("WETH", 0.0), 6),
            "wbtc_amount": round(b.collaterals.get("WBTC", 0.0), 6),
            "link_amount": round(b.collaterals.get("LINK", 0.0), 4),
            "collateral_value_usd": round(b.market_value(), 2),
            "debt_usd": round(b.debt, 2),
            "cash_usdc": round(b.cash_usdc, 2),
            "hf_before_liquidation": round(hf0, 4),
            "health_factor": round(hf, 4),
            "status": status,
            "liquidated_this_tick": len(evs),
            "debt_covered_this_tick_usd": round(sum(e["debt_covered"] for e in evs), 2),
            "seized_this_tick": "; ".join(f"{e['seized_asset']} {e['seized_amount']:.4f}" for e in evs),
            "n_liquidated_times": n_liq.get(b.unique_id, 0),
            "bad_debt_recorded_usd": round(b.bad_debt_recorded, 2),
        })
    return rows


def step_with_hf_before(model):
    """
    Giống DeFiContagionModel.step() nhưng chụp HF của mọi người vay ngay trước vòng thanh lý.
    Thứ tự giữ nguyên: tick + 1, sốc (tick 1), Borrower, [chụp HF], Liquidator, Arbitrageur, làm mới HF, DataCollector.
    (Phần đối chiếu cuối file xác nhận kết quả vẫn bằng model.step() gốc.)
    """
    model.current_tick += 1
    if model.current_tick == 1:
        model.inject_shock()
    for b in model.borrowers():
        b.step()
    hf_before = {b.unique_id: b.calculate_health_factor() for b in model.borrowers()}
    model.liquidator.step()
    model.arbitrageur.step()
    for b in model.borrowers():
        b.health_factor = b.calculate_health_factor()
    model.datacollector.collect(model)
    return hf_before


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--borrowers", type=int, default=60)
    ap.add_argument("--ticks", type=int, default=15)
    ap.add_argument("--shock-asset", default="WETH")
    ap.add_argument("--shock-pct", type=float, default=0.35)
    ap.add_argument("--depth", type=float, default=1.0)
    ap.add_argument("--close-factor", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oracle-mode", choices=("amm", "external"), default="amm")
    ap.add_argument("--n-liquidators", type=int, default=1)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--only-initial", action="store_true", help="chỉ xuất tick 0")
    args = ap.parse_args()

    model = DeFiContagionModel(
        n_borrowers=args.borrowers, shock_asset=args.shock_asset, shock_pct=args.shock_pct,
        amm_depth_multiplier=args.depth, close_factor=args.close_factor, seed=args.seed,
        oracle_mode=args.oracle_mode, n_liquidators=args.n_liquidators)

    initial = snapshot(model, 0)
    by_tick = list(initial)
    if not args.only_initial:
        for _ in range(args.ticks):
            hf_before = step_with_hf_before(model)
            by_tick.extend(snapshot(model, model.current_tick, hf_before))

    os.makedirs(args.out_dir, exist_ok=True)
    p0 = os.path.join(args.out_dir, "borrowers_tick0.csv")
    write_csv(p0, initial)
    print(f"Đã ghi {p0} ({len(initial)} dòng)")
    if not args.only_initial:
        p1 = os.path.join(args.out_dir, "borrowers_by_tick.csv")
        write_csv(p1, by_tick)
        print(f"Đã ghi {p1} ({len(by_tick)} dòng = {args.ticks + 1} tick x {len(initial)} người)")

    # ------------------------------------------------------------------ đối chiếu với thống kê
    print("\nĐỐI CHIẾU VỚI THỐNG KÊ CỦA MÔ HÌNH")
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'KHỚP' if ok else 'LỆCH'}] {name}" + (f"  ({detail})" if detail else ""))

    n = len(initial)
    exp = {0: round(0.4 * n), 2: round(0.2 * n)}
    exp[1] = max(0, n - exp[0] - exp[2])
    cnt = {g: sum(1 for r in initial if r["group_id"] == g) for g in GROUP_ASSETS}
    check("Số người mỗi nhóm = 40% / 40% / 20%", cnt == exp, f"{cnt[0]} / {cnt[1]} / {cnt[2]}")
    check("Giá trị thế chấp ban đầu trong khoảng quy định",
          all(MIN_COLLATERAL_USD * 0.999 <= r["collateral_value_usd"] <= MAX_COLLATERAL_USD * 1.001 for r in initial),
          f"{min(r['collateral_value_usd'] for r in initial):,.0f} đến {max(r['collateral_value_usd'] for r in initial):,.0f} USD")
    lo, hi = SEED_HF_RANGE
    check("HF ban đầu trong khoảng quy định",
          all(lo - 1e-3 <= r["health_factor"] <= hi + 1e-3 for r in initial),
          f"{min(r['health_factor'] for r in initial):.3f} đến {max(r['health_factor'] for r in initial):.3f}")
    n_active = sum(1 for r in initial if r["risk_profile"] == "active")
    check("Có người vay loại 'active' (mặc định khoảng 30%)", 0 < n_active < n, f"{n_active}/{n}")
    check("Nhóm 3 chỉ giữ LINK, nhóm 1 và 2 không giữ cùng bộ tài sản",
          all(r["weth_amount"] == 0 and r["wbtc_amount"] == 0 for r in initial if r["group_id"] == 2)
          and all(r["link_amount"] == 0 for r in initial if r["group_id"] == 0)
          and all(r["weth_amount"] == 0 for r in initial if r["group_id"] == 1))
    total_debt0 = sum(r["debt_usd"] for r in initial)
    print(f"  Tổng nợ ban đầu: {total_debt0:,.2f} USD; tổng thế chấp: "
          f"{sum(r['collateral_value_usd'] for r in initial):,.2f} USD")

    if not args.only_initial:
        last = [r for r in by_tick if r["tick"] == model.current_tick]
        check("CSV: tổng số lần thanh lý của 60 người = tổng vụ thanh lý của mô hình",
              sum(r["n_liquidated_times"] for r in last) == model.total_liquidations,
              f"{sum(r['n_liquidated_times'] for r in last)} và {model.total_liquidations}")
        check("CSV: số dòng nhật ký thanh lý = tổng vụ thanh lý",
              len(model.liquidation_events) == model.total_liquidations, f"{len(model.liquidation_events)}")
        check("CSV: số người đã bị thanh lý >= 1 lần = số người có is_liquidated",
              sum(1 for r in last if r["n_liquidated_times"] > 0) == sum(1 for b in model.borrowers() if b.is_liquidated),
              f"{sum(1 for r in last if r['n_liquidated_times'] > 0)} người")
        check("CSV: số dòng = (số tick + 1) x số người", len(by_tick) == (args.ticks + 1) * n)
        check("Mọi người bị thanh lý trong tick đều có HF < 1 ngay trước vòng thanh lý (tick 1 trở đi)",
              all(r["hf_before_liquidation"] < 1.0 for r in by_tick if r["liquidated_this_tick"] > 0 and r["tick"] == 1),
              "kiểm tại tick 1, khi chưa có thanh lý trước đó làm đổi giá")
        check("Số vụ thanh lý cộng theo cột liquidated_this_tick = tổng",
              sum(r["liquidated_this_tick"] for r in by_tick) == model.total_liquidations)
        print(f"  Cuối tick {model.current_tick}: tổng vụ thanh lý {model.total_liquidations}, "
              f"nợ xấu {model.bad_debt:,.2f} USD, "
              f"người có HF<1: {sum(1 for r in last if r['health_factor'] < 1 and r['debt_usd'] > 0)}")
        if (args.seed, args.shock_asset, args.shock_pct, args.depth, args.close_factor, args.borrowers,
                args.oracle_mode, args.n_liquidators, args.ticks) == (42, "WETH", 0.35, 1.0, 0.5, 60, "amm", 1, 15):
            check("Cấu hình mặc định: 185 vụ thanh lý, nợ xấu 291.012 USD (số tham chiếu)",
                  model.total_liquidations == 185 and round(model.bad_debt) == 291012,
                  f"{model.total_liquidations}; {model.bad_debt:,.0f}")

    print("\nMọi mục đều khớp." if ok_all else "\nCó mục LỆCH, kiểm tra lại tham số chạy.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
