"""
app.py - Giao diện tương tác Streamlit và các khung trực quan hóa.
Giao diện chỉ ĐỌC dữ liệu từ DataCollector/mô hình, không can thiệp logic tính toán.
"""
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import DeFiContagionModel
from run_experiment import indirect_share, mean_ci95
import run_stackelberg
from scenarios import SCENARIOS, amplification, events_frame, run_scenario

ASSETS = ["WETH", "WBTC", "LINK"]
COLOR_ASSET = "#38bdf8"
COLOR_SAFE = "#22c55e"
COLOR_DANGER = "#ef4444"
ASSET_COLORS = {"WETH": "#818cf8", "WBTC": "#f59e0b", "LINK": "#38bdf8"}


def _show(fn, obj) -> None:
    """Hiển thị chiếm hết chiều ngang, tương thích Streamlit cũ (use_container_width) và mới (width='stretch')."""
    try:
        fn(obj, width="stretch")
    except TypeError:
        fn(obj, use_container_width=True)


def render_sidebar_controls() -> dict:
    """Thanh công cụ bên trái để điều chỉnh tham số kịch bản. Trả về từ điển tham số."""
    st.sidebar.header("Tham số kịch bản")
    st.sidebar.caption("Các tham số dưới đây dùng chung cho cả hai kịch bản (cùng trạng thái ban đầu). "
                       "Mức sốc đặt riêng trong từng tab.")
    amm_depth = st.sidebar.select_slider("Độ sâu thanh khoản AMM (x)",
                                         options=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0], value=1.0)
    close_factor = st.sidebar.slider("Tỷ lệ nợ thanh lý (close factor)", 0.25, 1.0, 0.50, step=0.05)
    n_borrowers = st.sidebar.slider("Số lượng người vay", 30, 150, 60, step=10)
    n_ticks = st.sidebar.slider("Thời gian mô phỏng (ticks)", 10, 50, 15, step=1)
    seed = st.sidebar.number_input("Seed ngẫu nhiên", min_value=0, value=42, step=1)
    n_liq = st.sidebar.select_slider("Số Liquidator cạnh tranh gas", options=[1, 2, 5, 10], value=1)
    oracle_label = st.sidebar.radio(
        "Kênh oracle", ["Bám giá AMM (kênh lây lan BẬT)", "Giá ngoại sinh (kênh TẮT, đối chứng)"])
    return {
        "n_liquidators": int(n_liq),
        "oracle_mode": "amm" if oracle_label.startswith("Bám") else "external",
        "amm_depth_multiplier": amm_depth,
        "close_factor": close_factor,
        "n_borrowers": int(n_borrowers),
        "n_ticks": int(n_ticks),
        "seed": int(seed),
    }


def render_price_cascade_chart(df_metrics: pd.DataFrame) -> go.Figure:
    """Biểu đồ đa đường: giá oracle các tài sản theo % giá trị ban đầu qua từng tick."""
    fig = go.Figure()
    for asset in ASSETS:
        base = float(df_metrics[f"Price_{asset}"].iloc[0])
        fig.add_trace(go.Scatter(
            x=df_metrics.index, y=100.0 * df_metrics[f"Price_{asset}"] / base,
            mode="lines+markers", name=asset,
            line=dict(color=ASSET_COLORS[asset], width=2.5),
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Thác giá lan truyền (% so với giá ban đầu)",
        xaxis_title="Tick", yaxis_title="% giá ban đầu",
        hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_network_data(model) -> Tuple[nx.Graph, Dict[Any, Tuple[float, float]]]:
    """
    Dựng đồ thị lưỡng phân Tài sản - Người vay và toạ độ bố cục (thuần dữ liệu, không phụ thuộc plotly).
    Thuộc tính nút: kind ('asset'|'borrower'), color, size, label.
    """
    g = nx.Graph()
    for a in ASSETS:
        g.add_node(a, kind="asset", color=COLOR_ASSET, size=34, label=a)
    for b in model.borrowers():
        hf = b.calculate_health_factor()
        danger = hf < 1.0 or b.is_liquidated
        node = f"B{b.unique_id}"
        g.add_node(node, kind="borrower", color=COLOR_DANGER if danger else COLOR_SAFE, size=11,
                   label=f"{node} (nhóm {b.group_id + 1})<br>HF = {hf:.2f}<br>Nợ = ${b.debt:,.0f}")
        for asset, amount in b.collaterals.items():
            if asset in ASSETS:  # kể cả khi bị tịch thu hết, vẫn giữ cạnh lịch sử
                g.add_edge(node, asset)
    pos = nx.spring_layout(g, seed=7, k=0.9)
    return g, {n: (float(x), float(y)) for n, (x, y) in pos.items()}


def build_contagion_network_graph(model) -> go.Figure:
    """Đồ thị lưỡng phân (NetworkX + Plotly): nút tài sản xanh lam, người vay xanh lá/đỏ."""
    g, pos = build_network_data(model)
    ex: List[float] = []
    ey: List[float] = []
    for u, v in g.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none",
                            line=dict(width=0.6, color="rgba(148,163,184,0.5)"))
    nodes = list(g.nodes())
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=[n if g.nodes[n]["kind"] == "asset" else "" for n in nodes],
        textposition="middle center", textfont=dict(color="#0f172a", size=11),
        hovertext=[g.nodes[n]["label"] for n in nodes], hoverinfo="text",
        marker=dict(size=[g.nodes[n]["size"] for n in nodes],
                    color=[g.nodes[n]["color"] for n in nodes],
                    line=dict(width=1, color="#0f172a")),
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="Mạng lưới lây lan rủi ro (đỏ = vỡ nợ/đã bị thanh lý)", showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=50, b=10), height=520,
    )
    return fig


def run_scenario_ui(shocks: Dict[str, float], params: dict) -> dict:
    """Chạy MỘT kịch bản độc lập (mô hình mới, không dùng lại trạng thái của kịch bản khác)."""
    kw = {k: params[k] for k in ("n_borrowers", "amm_depth_multiplier", "close_factor", "seed",
                                 "oracle_mode", "n_liquidators")}
    model, dft = run_scenario(shocks, params["n_ticks"], **kw)
    amp = amplification(shocks, params["n_ticks"], **{**kw, "oracle_mode": "amm"})
    return {"model": model, "ticks": dft, "events": events_frame(model), "amp": amp, "shocks": dict(shocks)}


def render_scenario_cards(res: dict, single: bool) -> None:
    model, dft = res["model"], res["ticks"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Người vay bị thanh lý", f"{int(dft['cum_liquidated_borrowers'].iloc[-1])}/{len(model.borrowers())}")
    c2.metric("Tổng vụ thanh lý", f"{model.total_liquidations:,}")
    c3.metric("Giá trị tài sản bị thanh lý ($)", f"${dft['liquidated_value_usd'].sum():,.0f}")
    c4.metric("Tổng nợ xấu ($)", f"${model.bad_debt:,.0f}")
    if single:
        c5.metric("Tỷ trọng thanh lý gián tiếp", f"{100 * indirect_share(model):.1f}%")
    else:
        amp = res["amp"]
        c5.metric("Khuếch đại kênh AMM", "n/a" if amp == float("inf") else f"{amp:.2f}x")


def render_liquidation_per_tick_chart(dft: pd.DataFrame) -> go.Figure:
    d = dft[dft["tick"] >= 1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["tick"], y=d["liquidated_borrowers"], name="Người vay bị thanh lý",
                         marker_color=COLOR_DANGER))
    fig.add_trace(go.Scatter(x=d["tick"], y=d["cum_liquidated_borrowers"], name="Tích luỹ", mode="lines+markers",
                             line=dict(color="#f59e0b", width=2.5)))
    fig.update_layout(title="Người vay bị thanh lý qua từng tick (cột) và tích luỹ (đường)",
                      xaxis_title="Tick", yaxis_title="Số người vay", hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_hf_chart(dft: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dft["tick"], y=dft["avg_hf"], name="HF trung bình", mode="lines+markers",
                             line=dict(color=COLOR_SAFE, width=2.5)))
    fig.add_trace(go.Scatter(x=dft["tick"], y=dft["min_hf"], name="HF nhỏ nhất", mode="lines+markers",
                             line=dict(color=COLOR_DANGER, width=2.5)))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="HF = 1 (ngưỡng thanh lý)")
    fig.update_layout(title="Diễn biến Health Factor (người vay còn nợ)", xaxis_title="Tick",
                      yaxis_title="HF", hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_status_chart(dft: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("safe", "An toàn (HF >= 1,15 hoặc hết nợ)", COLOR_SAFE),
                             ("at_risk", "Rủi ro (1 <= HF < 1,15)", "#f59e0b"),
                             ("liquidatable", "Đủ điều kiện thanh lý (HF < 1)", COLOR_DANGER)):
        fig.add_trace(go.Bar(x=dft["tick"], y=dft[col], name=name, marker_color=color))
    fig.update_layout(barmode="stack", title="Số người vay an toàn / rủi ro / đủ điều kiện thanh lý theo tick",
                      xaxis_title="Tick", yaxis_title="Số người vay", hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_scenario_tab(key: str) -> None:
    """Một tab kịch bản: nhập mức sốc, chạy độc lập, hiển thị và xuất CSV. Trạng thái lưu theo key."""
    single = key == "single"
    defaults = SCENARIOS[key]
    if single:
        st.markdown("**Tab 1: chỉ một tài sản bị sốc tại tick 1; các tài sản khác và USDC giữ nguyên.**")
        c1, c2 = st.columns(2)
        asset = c1.selectbox("Tài sản bị sốc", ASSETS, index=0, key="single_asset")
        pct = c2.slider("Mức giảm giá (%)", 5, 70, int(100 * defaults["WETH"]), step=5, key="single_pct")
        shocks = {asset: pct / 100.0}
    else:
        st.markdown("**Tab 2: nhiều tài sản cùng giảm giá trong CÙNG tick 1; USDC giữ nguyên.**")
        cols = st.columns(3)
        shocks = {}
        for col, a in zip(cols, ASSETS):
            v = col.slider(f"{a} giảm (%)", 0, 50, int(100 * defaults[a]), step=1, key=f"multi_{a}")
            if v > 0:
                shocks[a] = v / 100.0
        st.caption(f"Tổng các tỷ lệ sốc đặt vào: {100 * sum(shocks.values()):.0f}% (mặc định 22 + 15 + 15 = 52%; "
                   "mức 15/10/10 không gây thanh lý nào ở seed 42). Đây là tổng tỷ lệ, không phải tổn thất danh mục.")
    params = st.session_state.get("params", {})
    label = "Khởi chạy Tab 1" if single else "Khởi chạy Tab 2"
    if st.button(label, type="primary", key=f"run_{key}"):
        if not shocks:
            st.warning("Cần ít nhất một tài sản có mức sốc > 0.")
        else:
            st.session_state[f"result_{key}"] = run_scenario_ui(shocks, params)
    res = st.session_state.get(f"result_{key}")
    if res is None:
        st.info("Chọn mức sốc rồi bấm nút khởi chạy. Hai tab chạy độc lập, không ảnh hưởng nhau.")
        return
    desc = ", ".join(f"{a} -{100 * v:.0f}%" for a, v in res["shocks"].items())
    st.caption(f"Kết quả đang hiển thị: {desc}.")
    model, dft = res["model"], res["ticks"]
    render_scenario_cards(res, single)
    if model.total_liquidations == 0:
        st.warning(f"Không có thanh lý nào: HF nhỏ nhất sau sốc là {dft['min_hf'].min():.2f} (>= 1), nên chưa người vay "
                   "nào chạm ngưỡng thanh lý và chuỗi lây lan qua AMM chưa được kích hoạt. "
                   "Tăng mức sốc để xem hiệu ứng ngưỡng.")
    _show(st.plotly_chart, render_price_cascade_chart(model.datacollector.get_model_vars_dataframe()))
    _show(st.plotly_chart, render_status_chart(dft))
    # Chỉ giữ 2 biểu đồ: thác giá và an toàn/rủi ro/đủ điều kiện thanh lý. Các hàm vẽ biểu đồ khác vẫn còn trong
    # file (render_liquidation_per_tick_chart, render_hf_chart, build_contagion_network_graph); muốn hiện lại chỉ cần gọi.
    st.subheader("Dữ liệu theo tick")
    _show(st.dataframe, dft)
    d1, d2 = st.columns(2)
    d1.download_button("Tải CSV theo tick", dft.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"scenario_{key}_ticks.csv", mime="text/csv", key=f"dl_t_{key}")
    d2.download_button("Tải CSV nhật ký thanh lý", res["events"].to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"scenario_{key}_events.csv", mime="text/csv", key=f"dl_e_{key}")
    st.subheader("Nhật ký thanh lý chi tiết")
    if len(res["events"]):
        _show(st.dataframe, res["events"])
    else:
        st.write("Không có thanh lý nào xảy ra.")


N_SEEDS_UI = 10     # cùng số seed với run_sensitivity.py và run_competition.py để số liệu khớp
N_TICKS_UI = 15
BASE_GAS = 20.0


@st.cache_data(show_spinner=False)
def compute_sensitivity(n_seeds: int = N_SEEDS_UI) -> pd.DataFrame:
    """Đổi MỘT tham số mỗi lần (độ sâu AMM, mức sốc), trả bảng trung bình ± CI95 qua nhiều seed."""
    sweeps = (("Độ sâu AMM (x)", "amm_depth_multiplier", (0.5, 1.0, 2.0)),
              ("Mức sốc WETH", "shock_pct", (0.25, 0.35, 0.50)))
    rows = []
    for label, key, values in sweeps:
        for v in values:
            shares, bads = [], []
            for seed in range(n_seeds):
                m = DeFiContagionModel(seed=seed, **{key: v})
                for _ in range(N_TICKS_UI):
                    m.step()
                shares.append(100 * indirect_share(m))
                bads.append(m.bad_debt / 1e3)
            (sh, sh_ci), (bd, bd_ci) = mean_ci95(shares), mean_ci95(bads)
            rows.append({"Tham số": label, "Giá trị": v, "Tỷ trọng gián tiếp (%)": sh, "CI95 tỷ trọng": sh_ci,
                         "Nợ xấu (nghìn $)": bd, "CI95 nợ xấu": bd_ci})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_competition(n_seeds: int = N_SEEDS_UI) -> pd.DataFrame:
    """Cạnh tranh gas: lợi nhuận TB mỗi khoản của người thắng so với dự đoán lý thuyết g/(n+1)."""
    rows = []
    for n in (1, 2, 5, 10):
        profit = fee = events = 0
        shares = []
        for seed in range(n_seeds):
            m = DeFiContagionModel(seed=seed, n_liquidators=n)
            for _ in range(N_TICKS_UI):
                m.step()
            ev = m.liquidation_events
            events += len(ev)
            profit += sum(e["profit"] for e in ev)
            fee += sum(e["priority_fee"] for e in ev)
            shares.append(100 * indirect_share(m))
        rows.append({"Số bot": n,
                     "Lợi nhuận TB/khoản (USD)": profit / events,
                     "Lý thuyết g/(n+1)": BASE_GAS / (n + 1) if n >= 2 else None,
                     "Phí ưu tiên / thặng dư (%)": 100 * fee / (fee + profit),
                     "Tỷ trọng gián tiếp (%)": mean_ci95(shares)[0]})
    return pd.DataFrame(rows)


def render_sensitivity_tab() -> None:
    st.subheader("Phân tích độ nhạy")
    st.caption(f"Mỗi lần đổi một tham số, giữ phần còn lại mặc định; {N_SEEDS_UI} seed, {N_TICKS_UI} tick. "
               "Câu hỏi: kết luận 'kênh oracle gây lây lan' có còn đúng khi đổi độ sâu AMM hoặc mức sốc?")
    if st.button("Chạy phân tích độ nhạy"):
        st.session_state["sens"] = compute_sensitivity()
    df = st.session_state.get("sens")
    if df is None:
        st.info("Bấm nút để chạy (khoảng vài chục giây).")
        return
    for param in df["Tham số"].unique():
        sub = df[df["Tham số"] == param]
        fig = go.Figure(go.Bar(x=sub["Giá trị"].astype(str), y=sub["Tỷ trọng gián tiếp (%)"],
                               error_y=dict(type="data", array=sub["CI95 tỷ trọng"]),
                               marker_color="#818cf8"))
        fig.update_layout(title=f"Tỷ trọng thanh lý gián tiếp theo {param}", xaxis_title=param,
                          yaxis_title="%", margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(type="category")
        _show(st.plotly_chart, fig)
    _show(st.dataframe, df.round({c: 1 for c in df.columns if c != "Giá trị"}))


def render_competition_tab() -> None:
    st.subheader("Nhiều Liquidator cạnh tranh gas (đấu giá giá tăng dần)")
    st.caption("Mỗi bot có chi phí gas riêng c_i ~ U(0.5g, 1.5g), g = 20 USD; bot thắng trả phí ưu tiên = G - c(2). "
               f"Dự đoán lý thuyết lợi nhuận người thắng = g/(n+1). {N_SEEDS_UI} seed, {N_TICKS_UI} tick.")
    if st.button("Chạy mô phỏng cạnh tranh gas"):
        st.session_state["comp"] = compute_competition()
    df = st.session_state.get("comp")
    if df is None:
        st.info("Bấm nút để chạy (khoảng vài chục giây).")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Số bot"].astype(str), y=df["Lợi nhuận TB/khoản (USD)"], name="Mô phỏng",
                         marker_color="#f59e0b"))
    fig.add_trace(go.Scatter(x=df["Số bot"].astype(str), y=df["Lý thuyết g/(n+1)"], name="Lý thuyết g/(n+1)",
                             mode="lines+markers", line=dict(color="#22c55e", width=2.5)))
    fig.update_layout(title="Lợi nhuận ròng mỗi khoản thanh lý của người thắng", xaxis_title="Số bot",
                      yaxis_title="USD", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(type="category")
    _show(st.plotly_chart, fig)
    _show(st.dataframe, df.round(2))
    st.caption("Đọc kết quả: thêm bot thì lợi nhuận người thắng giảm (phần còn lại trả thợ đào dưới dạng phí ưu tiên), "
               "còn tỷ trọng lây lan gián tiếp gần như không đổi.")


@st.cache_data(show_spinner="Đang chạy lưới Stackelberg (vài chục giây)...")
def compute_stackelberg(n_seeds: int = N_SEEDS_UI) -> pd.DataFrame:
    """Lưới (close factor, hệ số thưởng) x n_seeds seed; dùng đúng hàm của run_stackelberg.py."""
    return run_stackelberg.build_grid(n_seeds)


def render_stackelberg_tab() -> None:
    st.subheader("Trò chơi Stackelberg: giao thức chọn tham số, thanh lý viên phản ứng")
    st.caption("Leader (giao thức) chọn close factor kappa và hệ số thưởng s; follower (thanh lý viên) thanh lý khi có lời. "
               "Leader giải: min NợXấu + w x ThưởngĐãTrả, ràng buộc nợ tồn đọng <= rho và mức tham gia >= PART (thưởng quá thấp "
               "thì không ai thanh lý: nợ xấu = 0 nhưng rủi ro còn nguyên). Thanh lý viên theo quy tắc tham lam cố định, "
               f"không phải phản ứng chiến lược. {N_SEEDS_UI} seed, {N_TICKS_UI} tick, 1 thanh lý viên.")
    if st.button("Chạy lưới Stackelberg", key="run_stack"):
        st.session_state["stack"] = compute_stackelberg()
    grid = st.session_state.get("stack")
    if grid is None:
        st.info("Bấm nút để chạy (khoảng vài chục giây).")
        return
    metrics = {"Nợ xấu (USD)": "bad_debt", "Thưởng đã trả (USD)": "bonus_paid",
               "Nợ tồn đọng cuối kỳ (% nợ ban đầu)": "unresolved",
               "Mức tham gia (% khối lượng nợ thanh lý so với ô gốc)": "participation"}
    c1, c2, c3, c4 = st.columns(4)
    label = c1.selectbox("Chỉ số hiển thị", list(metrics), index=0, key="stack_metric")
    rho = c2.slider("Ngưỡng tồn đọng rho (%)", 5, 30, int(100 * run_stackelberg.RHO), step=5, key="stack_rho") / 100.0
    part = c3.slider("Tham gia tối thiểu PART (%)", 0, 100, int(100 * run_stackelberg.PART_MIN), step=25, key="stack_part") / 100.0
    w = c4.slider("Trọng số chi phí người dùng w", 0.0, 2.0, 1.0, step=0.5, key="stack_w")
    col = metrics[label]
    z = grid.pivot(index="kappa", columns="bonus_scale", values=col)
    if col in ("unresolved", "participation"):
        z = 100 * z
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
                               text=[[f"{v:,.0f}" if col not in ("unresolved", "participation") else f"{v:.1f}" for v in row] for row in z.values],
                               texttemplate="%{text}", colorscale="YlOrRd"))
    feasible = run_stackelberg.feasible_cells(grid, rho, part)
    if feasible.empty:
        st.warning("Không ô nào thỏa cả hai ràng buộc; tăng rho hoặc giảm PART.")
    else:
        loss = feasible["bad_debt"] + w * feasible["bonus_paid"]
        best = grid.loc[loss.idxmin()]
        fig.add_trace(go.Scatter(x=[str(best["bonus_scale"])], y=[str(best["kappa"])], mode="markers",
                                 marker=dict(symbol="star", size=22, color="#22c55e", line=dict(width=1, color="#0f172a")),
                                 name="Lựa chọn tối ưu của leader"))
        st.success(f"Lựa chọn tối ưu của giao thức (rho = {rho:.0%}, PART = {part:.0%}, w = {w}): kappa = {best['kappa']}, s = {best['bonus_scale']}; "
                   f"nợ xấu {best['bad_debt']:,.0f} USD, thưởng đã trả {best['bonus_paid']:,.0f} USD, "
                   f"tồn đọng {100 * best['unresolved']:.1f}%, tham gia {100 * best['participation']:.0f}%.")
    fig.update_layout(title=f"{label} theo (kappa, s); ngôi sao = tối ưu của leader", xaxis_title="Hệ số thưởng s",
                      yaxis_title="Close factor kappa", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    _show(st.plotly_chart, fig)
    base = grid[(grid.kappa == 0.5) & (grid.bonus_scale == 1.0)].iloc[0]
    st.caption(f"Tham số gốc (kappa = 0,5, s = 1): nợ xấu {base['bad_debt']:,.0f} USD, thưởng {base['bonus_paid']:,.0f} USD. "
               "Đọc kết quả: ở close factor >= 0,25, thưởng cao hơn đi kèm nợ xấu cao hơn (cơ chế chưa được tách: có thể do giá AMM "
               "rơi sâu hơn hoặc do HF của người vay giảm ngay sau thanh lý); nghiệm nằm ở góc lưới và đổi mạnh theo rho, "
               "PART, w, nên chỉ là kết luận có điều kiện của mô hình.")
    _show(st.dataframe, grid.round(3))


def main() -> None:
    """Hàm chính của ứng dụng Streamlit."""
    st.set_page_config(page_title="DeFi Contagion Simulator", layout="wide")
    st.title("Mô phỏng lây lan rủi ro thanh lý chéo DeFi")
    st.session_state["params"] = render_sidebar_controls()

    # Hai tab Độ nhạy và Đấu giá gas đang ẩn (render_sensitivity_tab / render_competition_tab và compute_* vẫn còn
    # trong file, check_consistency.py nhóm C vẫn dùng). Muốn bật lại: thêm tên tab vào st.tabs và gọi hàm render.
    tab_single, tab_multi, tab_stack = st.tabs(
        ["Tab 1: Sốc đơn tài sản", "Tab 2: Sốc đa tài sản", "Stackelberg"])
    with tab_single:
        render_scenario_tab("single")
    with tab_multi:
        render_scenario_tab("multi")
    with tab_stack:
        render_stackelberg_tab()


if __name__ == "__main__":
    main()
