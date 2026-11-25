"""
run_sensitivity.py - Phân tích độ nhạy: kết luận "kênh oracle bám AMM gây lây lan" có còn đúng
khi đổi độ sâu AMM hoặc mức sốc không?

Cách làm: mỗi lần chỉ đổi MỘT tham số, giữ các tham số khác ở mặc định, chạy N_SEEDS seed.
Chỉ số: tỷ trọng nợ thanh lý thuộc nhóm không giữ tài sản bị sốc (càng cao càng lây lan mạnh).

Chạy:  python run_sensitivity.py
"""
from model import DeFiContagionModel
from run_experiment import indirect_share, mean_ci95

N_SEEDS = 10
N_TICKS = 15
SWEEPS = {
    "amm_depth_multiplier": ("Độ sâu AMM (x)", (0.5, 1.0, 2.0)),
    "shock_pct": ("Mức sốc WETH", (0.25, 0.35, 0.50)),
}


def run_one(seed: int, **params) -> dict:
    model = DeFiContagionModel(seed=seed, **params)
    for _ in range(N_TICKS):
        model.step()
    return {"indirect_share": indirect_share(model), "bad_debt": model.bad_debt}


def main() -> None:
    print(f"{N_SEEDS} seed, {N_TICKS} tick/lần chạy\n")
    for param, (label, values) in SWEEPS.items():
        print(f"{label:<18}{'Tỷ trọng gián tiếp (%)':>26}{'Nợ xấu (nghìn $)':>22}")
        for v in values:
            runs = [run_one(s, **{param: v}) for s in range(N_SEEDS)]
            share, share_ci = mean_ci95([r["indirect_share"] * 100 for r in runs])
            bad, bad_ci = mean_ci95([r["bad_debt"] / 1e3 for r in runs])
            print(f"{v:<18}{share:>16.1f} ± {share_ci:<6.1f}{bad:>14.1f} ± {bad_ci:.1f}")
        print()


if __name__ == "__main__":
    main()
