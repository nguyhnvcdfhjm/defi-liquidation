# DeFi Lending ABM — Mô phỏng lây lan thanh lý bằng mô hình đa tác tử

Mô hình đa tác tử (Agent-Based Model, Mesa) của một thị trường cho vay DeFi, dùng để nghiên cứu **lây lan thanh lý qua giá AMM**: khi một cú sốc giá làm Liquidator xả tài sản thế chấp lên bể AMM, giá AMM giảm, oracle bám theo, và những người vay *không giữ tài sản bị sốc* cũng bị thanh lý.

Đề tài gồm ba phần:

| Phần | Nội dung | File chính |
|---|---|---|
| Mô hình nền | Borrower / Liquidator / Arbitrageur, 3 bể AMM, thanh lý, nợ xấu | `src/model.py`, `agents.py`, `amm.py`, `config.py` |
| Phần I | Sốc đơn tài sản (Tab 1) và sốc đa tài sản (Tab 2) | `src/scenarios.py`, `run_experiment.py`, `run_scenarios.py` |
| Phần II | Lý thuyết trò chơi: đấu giá phí ưu tiên, trò chơi giảm đòn bẩy của người vay, tối ưu tham số kiểu Stackelberg | `src/game_theory.py`, `run_stackelberg.py`, `run_game_compare.py` |

Báo cáo đầy đủ: [`docs/Bao_cao_NCKH_hoan_chinh.docx`](docs/Bao_cao_NCKH_hoan_chinh.docx).

## Cấu trúc thư mục

```
defi-lending-abm/
├── README.md
├── requirements.txt
├── LICENSE
├── src/                  Mã nguồn (chạy mọi lệnh từ thư mục này)
│   ├── model.py agents.py amm.py config.py     mô hình nền
│   ├── scenarios.py                            kịch bản Tab 1 / Tab 2
│   ├── game_theory.py                          module trò chơi người vay
│   ├── run_*.py                                các script thí nghiệm
│   ├── app.py                                  giao diện Streamlit
│   └── test_*.py                               50 kiểm thử (pytest)
├── results/              Kết quả đã chạy, dùng trong báo cáo
│   ├── tab1_single_shock/  tab2_multi_shock/  stackelberg/  game_theory/
│   └── tests.txt
├── figures/              Sáu hình dùng trong báo cáo và slide
└── docs/                 Báo cáo Word
```

## Cài đặt (Windows)

Đã chạy thử với Python 3.13 và Mesa 3.5.2; các phiên bản khác chưa được kiểm tra.

```bat
git clone <địa-chỉ-repo>
cd defi-lending-abm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd src
```

## Cách chạy

Tất cả lệnh dưới đây chạy trong thư mục `src/`.

| Mục đích | Lệnh | Kết quả đối chiếu |
|---|---|---|
| Kiểm thử | `python -m pytest -q` | 50 test đạt (`results/tests.txt`) |
| Tab 1: đối chứng kênh oracle–AMM, 30 seed | `python run_experiment.py` | `results/tab1_single_shock/tab1_run_experiment.txt` |
| Tab 1: độ nhạy | `python run_sensitivity.py` | `tab1_run_sensitivity.txt` |
| Tab 2: so sánh sốc đơn và sốc đa tài sản | `python run_scenarios.py` | `results/tab2_multi_shock/` |
| Tab 2: quét hệ số sốc λ, 30 seed | `python run_tab2_multiseed.py` | `tab2_multiseed.csv` |
| Cạnh tranh gas (đấu giá phí ưu tiên) | `python run_competition.py` | kết quả in ra màn hình; mô tả trong báo cáo |
| Stackelberg: lưới (κ, s), 10 seed | `python run_stackelberg.py` | `results/stackelberg/` |
| Trò chơi giảm đòn bẩy của người vay | `python run_game_compare.py` | `results/game_theory/` |
| Giao diện | `streamlit run app.py` | 3 tab: Sốc đơn, Sốc đa tài sản, Stackelberg |

Mỗi seed cho một quần thể 60 người vay khác nhau; các script dùng cùng dãy seed nên kết quả tái lập được.

## Kiến trúc mô hình (tóm tắt)

- **Tác tử:** 60 Borrower (3 nhóm: WETH+WBTC / WBTC+LINK / LINK), Liquidator (tham lam, tùy chọn nhiều bot cạnh tranh phí gas), Arbitrageur (kéo giá AMM về giá ngoại sinh).
- **Thị trường:** 3 bể AMM x·y=k (phí 0,3%); ba tầng giá: giá bể (`P_spot`), giá oracle (`P_oracle`), giá ngoại sinh (`P_ext`). `oracle_mode = "amm"` cho oracle bám giá bể; `"external"` tắt kênh lây lan (nhóm đối chứng).
- **Một tick (15 tick/lần chạy):** cú sốc (tick 1) → Borrower cập nhật HF → Liquidator chọn tài sản lợi nhuận Π cao nhất và chỉ thanh lý khi Π > 0 → Arbitrageur → thu số liệu.
- Chi tiết công thức, tham số và giả định: Chương 3 của báo cáo.

## Kết quả chính (đã chạy và kiểm chứng)

| Kết quả | Giá trị |
|---|---|
| Tab 1, seed 0–29: tỷ trọng thanh lý gián tiếp, kênh bật / tắt | 51,5 ± 0,8 % / 0 % |
| Tab 1, nợ xấu, kênh bật / tắt (nghìn USD) | 327,2 ± 14,1 / 0 |
| Tab 1 so với Tab 2 (λ = 1), seed 1–30: nợ xấu | 382,1 ± 14,1 so với 328,8 ± 14,5 (nghìn USD) |
| Tab 2: ngưỡng kích hoạt | λ = 0,7: 0/30 seed; λ = 0,85: 22/30; λ ≥ 0,9: 30/30 |
| Stackelberg (ρ = 15 %, PART = 0,5, 10 seed) | chỉ 3 ô lưới thỏa ràng buộc; ô gốc không thỏa |

## Giới hạn cần biết

- Oracle bám giá AMM là **giả định** của mô hình, không phải mô tả giao thức thực.
- Quần thể người vay được sinh theo tham số tự chọn, chưa đối chiếu dữ liệu on-chain.
- Follower trong phần Stackelberg là Liquidator tham lam cố định, nên kết quả là tối ưu của leader trên mô phỏng, chưa phải cân bằng Stackelberg chặt chẽ.
- Monte Carlo với giá lịch sử (`run_montecarlo_historical.py`) ; trò chơi người vay có nhiều cân bằng (mô tả trong Phần II của báo cáo).
- Hiệu chỉnh tham số theo Aave (`calibration.py`) chưa được dùng trong các kết quả trên.

## Thành viên

| Họ tên | MSSV | Phần phụ trách |
|---|---|---|
| Nguyễn Thị Yến | B23DCTC126 |  |
| Trần Thùy Trang | B23DCTC110 |  |
| Nguyễn Tống Xuân Mai | B23DCTC060 |  |


