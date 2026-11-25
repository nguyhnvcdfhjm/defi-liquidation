"""
calibration.py - Hiệu chỉnh từ dữ liệu công bố: (1) giá lịch sử -> phân phối cú sốc, (2) tham số giao thức.

Phần 1. Giá lịch sử (file CSV có cột date, close; tải bằng fetch_prices.py)
  Cú sốc h ngày tại thời điểm t:  drop_h(t) = 1 - P(t+h) / P(t).
  drop > 0: giá giảm; drop <= 0: giá tăng (không gây thanh lý, mô hình không chạy).
  Mô hình ABM áp cú sốc tức thời tại tick 1, nên h là "độ dài cửa sổ" mà ta coi là tức thời so với
  tốc độ phản ứng của thị trường. Đây là một GIẢ ĐỊNH và được đưa vào phân tích độ nhạy theo h.

Phần 2. Tham số giao thức công bố (PUBLISHED_PARAMS)
  Nguồn: bảng "Initial Set Results" trong đề xuất của Chaos Labs, "Aave V3 Ethereum (Initial
  Parameters)", diễn đàn quản trị Aave (governance.aave.com/t/.../11159). Giá trị trên chuỗi hiện tại
  có thể đã được quản trị thay đổi sau đề xuất đó, nên đây là cấu hình THAM CHIẾU, không phải trạng thái
  hiện hành. Close factor 50% (100% khi sức khoẻ vị thế thấp hơn ngưỡng) theo tài liệu nhà phát triển
  Aave, docs.aave.com/developers/guides/liquidations; ngưỡng HF chính xác không kiểm chứng được nên
  mô hình giữ close factor cố định 50%.
"""
import csv
from typing import Dict, List

PUBLISHED_PARAMS: Dict[str, Dict[str, float]] = {
    "liquidation_thresholds": {"WETH": 0.825, "WBTC": 0.75, "LINK": 0.65},
    "liquidation_bonuses": {"WETH": 0.05, "WBTC": 0.0625, "LINK": 0.075},
}


def load_prices(path: str) -> List[float]:
    """Đọc cột close (giá đóng cửa theo ngày) theo thứ tự thời gian."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "close" not in rows[0]:
        raise ValueError("CSV cần có cột 'close' (và 'date')")
    if "date" in rows[0]:
        rows.sort(key=lambda r: r["date"])
    prices = [float(r["close"]) for r in rows]
    if any(p <= 0 for p in prices):
        raise ValueError("Có giá <= 0 trong dữ liệu")
    return prices


def h_day_drops(prices: List[float], h: int) -> List[float]:
    """drop_h(t) = 1 - P(t+h)/P(t) cho mọi cửa sổ trượt h ngày (cửa sổ chồng lấn)."""
    if h < 1 or len(prices) <= h:
        raise ValueError("Cần h >= 1 và nhiều hơn h quan sát")
    return [1.0 - prices[t + h] / prices[t] for t in range(len(prices) - h)]


def quantile(sorted_values: List[float], q: float) -> float:
    """Phân vị q (0..1) của danh sách đã sắp xếp, nội suy tuyến tính."""
    if not sorted_values:
        raise ValueError("Danh sách rỗng")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def var_es(losses: List[float], alpha: float = 0.95):
    """VaR và ES (kỳ vọng tổn thất có điều kiện vượt VaR) mức alpha của một mẫu tổn thất."""
    s = sorted(losses)
    var = quantile(s, alpha)
    k = min(len(s) - 1, int(alpha * len(s)))   # ES = trung bình của (1-alpha) phần lớn nhất
    tail = s[k:]
    return var, sum(tail) / len(tail)


def worst_episodes(dates: List[str], prices: List[float], h: int, k: int = 3):
    """k cửa sổ h ngày giảm mạnh nhất, không chồng lấn: [(ngày bắt đầu, ngày kết thúc, mức giảm)]."""
    drops = h_day_drops(prices, h)
    order = sorted(range(len(drops)), key=lambda t: drops[t], reverse=True)
    chosen: List[int] = []
    for t in order:
        if all(abs(t - u) >= h for u in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    return [(dates[t], dates[t + h], drops[t]) for t in chosen]


def load_dates(path: str) -> List[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["date"])
    return [r["date"] for r in rows]
