"""
run_competition.py - Cạnh tranh gas giữa nhiều Liquidator (lý thuyết trò chơi mức đơn giản).

Mô hình trò chơi (xem english_auction trong agents.py):
    - Người chơi : n bot Liquidator. Cùng thấy một khoản thanh lý có lợi nhuận gộp G.
    - Chiến lược : mức phí ưu tiên (PGA). Bot i có chi phí gas riêng c_i ~ U(0.5g, 1.5g), g = 20 USD.
    - Quy tắc    : đấu giá giá tăng dần, ai trả phí cao nhất thì được thanh lý.
    - Cân bằng   : mỗi bot tăng phí tới khi bằng giá trị v_i = G - c_i của mình (chiến lược trội).
                   Người thắng = bot có c_i thấp nhất, phí ưu tiên = G - c_(2),
                   lợi nhuận ròng của người thắng = c_(2) - c_(1).
    - Dự đoán lý thuyết (n >= 2, G đủ lớn): chênh lệch hai chi phí thấp nhất của n biến đều U(a, b)
      có kỳ vọng (b - a)/(n + 1) = g/(n + 1)   [b - a = g].
      => lợi nhuận kỳ vọng mỗi khoản thanh lý của người thắng = g/(n + 1), GIẢM khi n tăng.

Câu hỏi: (1) số liệu mô phỏng có khớp dự đoán g/(n+1) không?
         (2) nhiều bot cạnh tranh có chặn được lây lan không?

Chạy:  python run_competition.py
Lưu ý: đây là đấu giá một vòng, thông tin riêng về chi phí; KHÔNG phải cân bằng hoàn hảo từng trò chơi con
(SPE) của một trò chơi nhiều giai đoạn.
"""
from model import DeFiContagionModel
from run_experiment import indirect_share, mean_ci95

N_LIST = (1, 2, 5, 10)   # số Liquidator
N_SEEDS = 10
N_TICKS = 15
BASE_GAS = 20.0          # g (USD), khớp MarketConfig.default_gas_cost_usd


def run_one(n_liquidators: int, seed: int) -> dict:
    model = DeFiContagionModel(seed=seed, n_liquidators=n_liquidators)
    for _ in range(N_TICKS):
        model.step()
    ev = model.liquidation_events
    return {
        "n_events": len(ev),
        "profit_sum": sum(e["profit"] for e in ev),
        "fee_sum": sum(e["priority_fee"] for e in ev),
        "indirect_share": indirect_share(model),
        "bad_debt": model.bad_debt,
    }


def main() -> None:
    print(f"{N_SEEDS} seed, {N_TICKS} tick, g = {BASE_GAS:.0f} USD\n")
    head = (f"{'n bot':>6}{'LN TB/khoản (USD)':>20}{'Lý thuyết g/(n+1)':>20}"
            f"{'Phí ưu tiên/thặng dư':>23}{'Tỷ trọng gián tiếp %':>23}")
    print(head)
    for n in N_LIST:
        runs = [run_one(n, s) for s in range(N_SEEDS)]
        events = sum(r["n_events"] for r in runs)
        profit = sum(r["profit_sum"] for r in runs) / events      # gộp mọi seed
        fee = sum(r["fee_sum"] for r in runs)
        surplus = fee + sum(r["profit_sum"] for r in runs)        # thặng dư trước khi trả phí ưu tiên
        theory = f"{BASE_GAS / (n + 1):,.2f}" if n >= 2 else "-"
        share, ci = mean_ci95([r["indirect_share"] * 100 for r in runs])
        print(f"{n:>6}{profit:>20,.2f}{theory:>20}{100 * fee / surplus:>22.1f}%"
              f"{share:>16.1f} ± {ci:.1f}")
    print("\nĐọc kết quả: lợi nhuận ròng mỗi khoản của người thắng giảm khi có thêm bot (phần còn lại "
          "trả cho thợ đào dưới dạng phí ưu tiên), còn tỷ trọng lây lan gián tiếp gần như không đổi.")


if __name__ == "__main__":
    main()
