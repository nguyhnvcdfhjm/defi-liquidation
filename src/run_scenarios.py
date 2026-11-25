"""
run_scenarios.py - Chạy cả hai kịch bản (sốc đơn / sốc đa tài sản), in so sánh và xuất CSV.

Chạy:  python run_scenarios.py [--seed 42] [--ticks 15] [--out-dir .]
Ra:    scenario_single_ticks.csv, scenario_single_events.csv, scenario_multi_ticks.csv, scenario_multi_events.csv
"""
import argparse
import os

from scenarios import SCENARIO_LABELS, SCENARIOS, amplification, events_frame, run_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=15)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"seed {a.seed}, {a.ticks} tick, cùng quần thể người vay cho hai kịch bản\n")
    results = {}
    for key, shocks in SCENARIOS.items():
        model, df = run_scenario(shocks, a.ticks, seed=a.seed)
        df.to_csv(os.path.join(a.out_dir, f"scenario_{key}_ticks.csv"), index=False, encoding="utf-8-sig")
        events_frame(model).to_csv(os.path.join(a.out_dir, f"scenario_{key}_events.csv"),
                                   index=False, encoding="utf-8-sig")
        amp = amplification(shocks, a.ticks, seed=a.seed)
        results[key] = (model, df, amp)
        desc = ", ".join(f"{k} -{v:.0%}" for k, v in shocks.items())
        print(f"[{SCENARIO_LABELS[key]}] {desc}")
        print(f"  Người vay bị thanh lý (tích luỹ): {df['cum_liquidated_borrowers'].iloc[-1]}")
        print(f"  Vụ thanh lý: {len(model.liquidation_events)} | giá trị thế chấp bị tịch thu: "
              f"{df['liquidated_value_usd'].sum():,.0f} USD | nợ xấu: {model.bad_debt:,.0f} USD")
        print(f"  Cuối kỳ: an toàn {df['safe'].iloc[-1]}, rủi ro {df['at_risk'].iloc[-1]}, "
              f"HF<1 {df['liquidatable'].iloc[-1]}; HF nhỏ nhất {df['min_hf'].iloc[-1]:.2f}")
        print(f"  Hệ số khuếch đại kênh AMM (nợ thanh lý bật/tắt kênh): {amp:.2f}x\n")
        print("  tick: người vay bị thanh lý mỗi tick -> " +
              " ".join(str(int(x)) for x in df["liquidated_borrowers"].iloc[1:]))
        print()
    # Quét hệ số nhân λ của cú sốc đa tài sản: tìm ngưỡng kích hoạt chuỗi thanh lý.
    print("Quét hệ số nhân λ cho sốc đa tài sản (λ = 1 là 22/15/15; λ ≈ 0,68 là 15/10/10):")
    print(f"  {'λ':<6}{'WETH/WBTC/LINK':<18}{'Người vay bị TL':>16}{'Vụ TL':>8}{'Nợ xấu ($)':>12}{'HF min tick 1':>15}")
    for lam in (0.5, 0.7, 0.85, 1.0, 1.25):
        sh = {k: min(0.95, v * lam) for k, v in SCENARIOS["multi"].items()}
        m, d = run_scenario(sh, a.ticks, seed=a.seed)
        tag = "/".join(f"{100 * v:.0f}" for v in sh.values())
        print(f"  {lam:<6}{tag:<18}{int(d['cum_liquidated_borrowers'].iloc[-1]):>16}{len(m.liquidation_events):>8}"
              f"{m.bad_debt:>12,.0f}{d['min_hf'].iloc[1]:>15.2f}")
    print()
    print(f"Đã lưu 4 file CSV vào '{a.out_dir}'.")


if __name__ == "__main__":
    main()
