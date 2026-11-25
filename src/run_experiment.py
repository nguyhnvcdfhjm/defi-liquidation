"""
run_experiment.py - Thí nghiệm đối chứng duy nhất: kênh lây lan BẬT so với TẮT.

Câu hỏi nghiên cứu:
    Nhóm người vay KHÔNG giữ tài sản bị sốc có bị thanh lý chỉ vì kênh
    "xả bán trên AMM -> trượt giá -> oracle giảm -> HF giảm" hay không?

Thiết kế:
    - Nhóm thực nghiệm : oracle_mode = "amm"      (oracle bám giá AMM, kênh lây lan bật)
    - Nhóm đối chứng   : oracle_mode = "external" (oracle chỉ theo giá ngoại sinh, kênh tắt)
    - Mọi thứ khác GIỐNG NHAU. Mỗi nhóm chạy 30 seed (30 quần thể người vay khác nhau),
      dùng cùng danh sách seed cho cả hai nhóm (so sánh cặp).
    - Chỉ số chính: tỷ trọng nợ bị thanh lý thuộc nhóm KHÔNG giữ tài sản bị sốc.
      (Nhóm không giữ tài sản sốc chỉ có thể bị thanh lý qua kênh lây lan.)

Chạy:  python run_experiment.py
"""
import math
import statistics

from model import DeFiContagionModel, GROUP_ASSETS

N_SEEDS = 30
N_TICKS = 15


def indirect_share(model) -> float:
    """Tỷ trọng nợ đã thanh lý thuộc nhóm KHÔNG giữ tài sản bị sốc (0 đến 1)."""
    # Nhóm trực tiếp = nhóm có giữ tài sản bị sốc (với WETH: Nhóm 1 = group_id 0).
    direct = {g for g, assets in GROUP_ASSETS.items() if model.shock_asset in assets}
    total = sum(e["debt_covered"] for e in model.liquidation_events)
    indirect = sum(e["debt_covered"] for e in model.liquidation_events
                   if e["group_id"] not in direct)
    return indirect / total if total > 0 else 0.0


def run_one(seed: int, oracle_mode: str) -> dict:
    """Chạy một mô phỏng và trả về các chỉ số cần so sánh."""
    model = DeFiContagionModel(seed=seed, oracle_mode=oracle_mode)
    for _ in range(N_TICKS):
        model.step()
    return {
        "indirect_share": indirect_share(model),
        "liquidated_debt": sum(e["debt_covered"] for e in model.liquidation_events),
        "bad_debt": model.bad_debt,
    }


def mean_ci95(values):
    """Trung bình và nửa độ rộng khoảng tin cậy 95% (xấp xỉ chuẩn: 1.96 * sd / sqrt(n))."""
    m = statistics.mean(values)
    ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return m, ci


def main() -> None:
    seeds = range(N_SEEDS)
    results = {mode: [run_one(s, mode) for s in seeds] for mode in ("amm", "external")}

    print(f"{N_SEEDS} seed, {N_TICKS} tick/lần chạy, cú sốc WETH -35%\n")
    print(f"{'Chỉ số':<34}{'Kênh BẬT (amm)':>20}{'Kênh TẮT (external)':>22}")
    for key, label, scale in (
            ("indirect_share", "Tỷ trọng nợ thanh lý gián tiếp (%)", 100),
            ("liquidated_debt", "Nợ đã thanh lý (nghìn $)", 1e-3),
            ("bad_debt", "Nợ xấu (nghìn $)", 1e-3)):
        cells = []
        for mode in ("amm", "external"):
            m, ci = mean_ci95([r[key] for r in results[mode]])
            cells.append(f"{m * scale:,.1f} ± {ci * scale:,.1f}")
        print(f"{label:<34}{cells[0]:>20}{cells[1]:>22}")

    n_on = sum(r["indirect_share"] > 0 for r in results["amm"])
    n_off = sum(r["indirect_share"] > 0 for r in results["external"])
    print(f"\nSố seed có thanh lý gián tiếp: kênh bật {n_on}/{N_SEEDS}, kênh tắt {n_off}/{N_SEEDS}")


if __name__ == "__main__":
    main()
