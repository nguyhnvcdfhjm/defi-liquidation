"""
fetch_prices.py - Tải giá đóng cửa ngày của ETH, BTC, LINK (so với USDT) từ API công khai của Binance.

Ra 3 file: prices_WETH.csv, prices_WBTC.csv, prices_LINK.csv (cột date,close). ETH/BTC/LINK là giá của
tài sản gốc; WBTC và wrapped token bám giá BTC gốc nên dùng làm đại diện (ghi chú này vào báo cáo).
CHẠY TRÊN MÁY CỦA BẠN (cần Internet):   python fetch_prices.py
Không cần khoá API, chỉ dùng thư viện chuẩn. Binance trả tối đa 1000 nến mỗi lần gọi nên script phân trang.
Nếu bị chặn (một số vùng/mạng), tải CSV giá ETH theo ngày từ nguồn khác (CoinGecko, Yahoo Finance,
CryptoDataDownload...) rồi đổi tên cột thành date,close; các script sau chỉ cần file đó.
"""
import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

URL = "https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&startTime={start}&limit=1000"


def fetch(symbol: str, start_ms: int):
    rows = []
    while True:
        with urllib.request.urlopen(URL.format(sym=symbol, start=start_ms), timeout=30) as r:
            batch = json.load(r)
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    return rows


PAIRS = {"WETH": "ETHUSDT", "WBTC": "BTCUSDT", "LINK": "LINKUSDT"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01", help="YYYY-MM-DD")
    a = ap.parse_args()
    start_ms = int(datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    for asset, symbol in PAIRS.items():
        rows = fetch(symbol, start_ms)
        out = f"prices_{asset}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "close"])
            for k in rows:
                w.writerow([datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"), k[4]])
        print(f"Đã lưu {len(rows)} dòng vào {out} ({symbol}, từ {a.start})")


if __name__ == "__main__":
    main()
