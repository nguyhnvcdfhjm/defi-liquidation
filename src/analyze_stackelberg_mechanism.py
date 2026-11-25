"""
analyze_stackelberg_mechanism.py - Kiểm tra giả thuyết cơ chế sau kết quả "thưởng cao hơn thì nợ xấu cao hơn".

Hai giải thích cạnh tranh (chưa ai tách trước đây):
  (A) Qua giá AMM: thanh lý bán nhiều tài sản thế chấp hơn vào AMM, giá rơi sâu hơn.
  (B) Trực tiếp qua HF: một khoản thanh lý trả D nợ và lấy D*(1+thưởng) thế chấp; HF mới cao hơn HF cũ chỉ khi
      LT*(1+thưởng) < HF cũ. Thưởng đủ lớn thì thanh lý làm HF của chính người vay GIẢM.
Kết quả (xem out/stackelberg_mechanism.csv): phần lớn người vay bị thanh lý đã rất sâu dưới nước (HF trước ~0,5-0,6 sau
cú sốc -35%); với nhóm này thanh lý làm HF GIẢM (thế chấp bị lấy nhanh hơn nợ giảm), và tỷ lệ HF về 0 tăng theo thưởng.
Với nhóm HF trước >= 0,9, thanh lý làm HF tăng. Đây là bằng chứng mô tả, KHÔNG phải thí nghiệm phản thực tế tách (A)
khỏi (B); muốn tách cần tắt kênh giá AMM trong một mô hình đối chứng.
Script chỉ ĐO (không sửa model/agents): bọc calculate_health_factor để ghi HF trước/sau mỗi khoản thanh lý
(HF sau tính trước khi oracle cập nhật theo giá AMM sau cú bán, nên tách được hiệu ứng (B) khỏi (A)).
Chạy: python analyze_stackelberg_mechanism.py   -> out/stackelberg_mechanism.csv
"""
import collections
import csv
import os
import sys

import agents
from model import DeFiContagionModel
import run_stackelberg as st

SCALES = (0.5, 1.0, 1.5, 2.0)
KAPPA = 0.5


def one(seed: int, scale: float):
    """Chạy một lần; trả về danh sách (HF_trước, HF_sau) cho từng khoản thanh lý."""
    base = DeFiContagionModel(seed=seed).liquidation_bonuses
    m = DeFiContagionModel(seed=seed, close_factor=KAPPA, liquidation_bonuses={a: b * scale for a, b in base.items()})
    hist = collections.defaultdict(list)
    last = {"uid": None}
    pairs = []
    orig_calc = agents.BorrowerAgent.calculate_health_factor
    orig_sync = m.sync_oracle

    def calc(self):
        v = orig_calc(self)
        hist[self.unique_id].append(v)
        last["uid"] = self.unique_id
        return v

    def sync(asset, price):
        # Trong agents.py: HF mới được gán NGAY TRƯỚC sync_oracle, và HF cũ được tính ngay trước khi thanh lý.
        h = hist[last["uid"]]
        if isinstance(sys._getframe(1).f_locals.get("self"), agents.LiquidatorAgent) and len(h) >= 2:
            pairs.append((h[-2], h[-1]))
        return orig_sync(asset, price)

    agents.BorrowerAgent.calculate_health_factor = calc
    m.sync_oracle = sync
    try:
        for _ in range(st.N_TICKS):
            m.step()
    finally:
        agents.BorrowerAgent.calculate_health_factor = orig_calc
    assert len(pairs) == len(m.liquidation_events), "ghép cặp HF không khớp số khoản thanh lý"
    return pairs


def main() -> None:
    os.makedirs("out", exist_ok=True)
    rows = []
    for scale in SCALES:
        pairs = [p for seed in range(st.N_SEEDS) for p in one(seed, scale)]
        n = len(pairs)
        hi = [(a, b) for a, b in pairs if a >= 0.9]
        lo = [(a, b) for a, b in pairs if a < 0.9]
        rows.append({
            "kappa": KAPPA, "bonus_scale": scale, "so_khoan_thanh_ly": n,
            "HF_truoc_trung_binh": round(sum(a for a, _ in pairs) / n, 3),
            "HF_sau_trung_binh": round(sum(b for _, b in pairs) / n, 3),
            "ty_le_HF_sau_bang_0": round(sum(b < 1e-9 for _, b in pairs) / n, 3),
            "so_khoan_HF_truoc>=0.9": len(hi),
            "ty_le_HF_giam_khi_HF_truoc>=0.9": round(sum(b < a for a, b in hi) / max(1, len(hi)), 3),
            "so_khoan_HF_truoc<0.9": len(lo),
            "ty_le_HF_giam_khi_HF_truoc<0.9": round(sum(b < a for a, b in lo) / max(1, len(lo)), 3)})
        print(rows[-1])
    with open("out/stackelberg_mechanism.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
