"""
game_theory.py - MODULE BỔ SUNG: trò chơi giảm đòn bẩy (deleveraging game) giữa những người vay.

Vì sao cần module này
---------------------
Trong mô hình gốc, người vay KHÔNG có chiến lược: "active" trả nợ theo luật cố định khi 1 < HF < 1,15,
"passive" không làm gì. Phần lý thuyết trò chơi hiện có chỉ nằm ở phía Liquidator (đấu giá phí ưu tiên)
và phía giao thức (Stackelberg). Module này thêm phía còn thiếu: người vay là người chơi có lý trí.

Mô hình trò chơi
----------------
- Người chơi : các người vay đủ điều kiện (mặc định chỉ nhóm "active", để so sánh cùng đối tượng).
- Chiến lược : tỷ lệ nợ trả trước x_i thuộc lưới {0, ..., x_max}, giới hạn bởi tiền mặt.
- Thông tin  : mọi người biết cơ chế thị trường (kỳ vọng hợp lý) và dự báo "đáy giá" của đợt thanh lý
               bằng cách chạy thử H tick tới trong một hộp cát (sandbox) rồi hoàn nguyên trạng thái.
- Hàm lợi ích (chi phí kỳ vọng, người vay tối thiểu hóa):
      C_i(x_i | giá đáy) = P_liq(HF_dự_báo) * Thiệt_hại_thanh_lý + phi * x_i * D_i
      HF_dự_báo = sum(thế chấp * giá_đáy * LT) / (D_i * (1 - x_i))
      P_liq     = 1 / (1 + exp((HF_dự_báo - 1) / tau))        (bất định về đường giá, độ rộng tau)
      Thiệt_hại = close_factor * D_i * (1 - x_i) * thưởng_thanh_lý_bình_quân_theo_tỷ_trọng_thế_chấp
- Ngoại ứng : mỗi người trả nợ làm bớt xả bán trên AMM nên giá đáy cao hơn và người khác ít rủi ro hơn
              (hàng hóa công ổn định giá). Có hai lực ngược nhau: (i) free-rider ở biên, người khác trả nợ
              thì mình đỡ phải trả; (ii) bổ trợ chiến lược: nếu không ai trả, đáy giá rất sâu và khoản trả
              tối đa 20% nợ của riêng mình VÔ ÍCH (HF dự báo vẫn << 1) nên không ai muốn trả. Do (ii) trò chơi
              có thể có NHIỀU cân bằng: "cú thác" (không ai trả) và "không thác" (đa số trả). Tập tham số phi
              ở giữa cho thấy hiện tượng này (xem bảng độ nhạy).
- Cân bằng  : hồ sơ x mà mọi người đã đáp trả tốt nhất với đáy giá do chính hồ sơ ấy sinh ra (trò chơi nhiều
              người nhỏ, price-taking). Giải bằng fictitious play xuất phát từ niềm tin bi quan (không ai trả
              nợ). Báo cáo epsilon = hối tiếc lớn nhất còn lại (tỷ lệ nợ) và cờ hội tụ; epsilon = 0 là cân
              bằng Nash trên lưới. Nghiệm trả về là cân bằng đạt được từ niềm tin bi quan, KHÔNG khẳng định duy nhất.
- Thời gian : cuốn chiếu (receding horizon): mỗi tick chơi lại với thông tin mới, không cam kết tương lai.

Hai chế độ
----------
- mode="nash"    : mỗi người tự tối ưu (trò chơi không hợp tác).
- mode="planner" : một nhà hoạch định xã hội chọn CHUNG một tỷ lệ x cho các người chơi đang rủi ro, tối thiểu
                   tổng thiệt hại xã hội = thưởng thanh lý người vay mất + nợ xấu + phí cơ hội trả nợ.
                   Dùng làm mốc để đo "giá của sự vô chính phủ" (price of anarchy) của chế độ nash.

Giới hạn (cần nói thẳng)
------------------------
- Người vay đã HF < 1 không được chữa (allow_cure=False): bot thanh lý thắng cuộc đua giao dịch.
- Đáy giá dùng min theo từng tài sản trong cả H tick (thận trọng, các đáy không nhất thiết đồng thời).
- Người chơi chỉ lên kế hoạch trả nợ; chưa có chiến lược bán thế chấp, thêm thế chấp, hay vay lại.
- phi, tau là tham số giả định (không hiệu chỉnh từ dữ liệu); có quét độ nhạy theo phi.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from model import DeFiContagionModel
from agents import BorrowerAgent

ASSETS = ("WETH", "WBTC", "LINK")


@dataclass
class GameConfig:
    """Tham số của module. Mọi giá trị đều là giả định có thể quét độ nhạy."""
    mode: str = "nash"               # "nash" | "planner"
    players: str = "active"          # "active" (24/60, như luật cũ) | "all"
    phi: float = 0.03                # chi phí cơ hội trả nợ trước, tính trên số tiền trả
    tau: float = 0.03                # độ rộng bất định của HF dự báo (logistic)
    horizon: int = 4                 # số tick chạy thử khi dự báo đáy giá
    max_repay_frac: float = 0.20     # trần x, bằng trần của luật cũ (20% nợ)
    n_actions: int = 5               # số điểm lưới của x, gồm cả 0
    max_iter: int = 12               # số vòng fictitious play tối đa mỗi tick
    allow_cure: bool = False         # cho phép chữa khi HF < 1 (mặc định tắt)
    risk_band: float = 1.10          # planner: coi là "đang rủi ro" nếu HF dự báo (x=0) < ngưỡng này
    liq_extra_penalty: float = 0.0   # thiệt hại cố định thêm khi bị thanh lý, tỷ lệ theo nợ (gas, phí)
    planner_sweeps: int = 1          # planner: số lượt tối ưu tọa độ theo từng người sau khi chọn x chung

    def __post_init__(self) -> None:
        if self.mode not in ("nash", "planner"):
            raise ValueError("mode phải là 'nash' hoặc 'planner'")
        if self.players not in ("active", "all"):
            raise ValueError("players phải là 'active' hoặc 'all'")
        if self.n_actions < 2 or self.horizon < 1 or self.max_iter < 1:
            raise ValueError("n_actions >= 2, horizon >= 1, max_iter >= 1")
        if not 0.0 < self.max_repay_frac < 1.0 or self.tau <= 0 or self.phi < 0:
            raise ValueError("tham số ngoài miền hợp lệ")

    @property
    def grid(self) -> List[float]:
        return [self.max_repay_frac * i / (self.n_actions - 1) for i in range(self.n_actions)]


# --------------------------------------------------------------------------- hộp cát
@contextmanager
def sandbox(model: DeFiContagionModel):
    """Chụp toàn bộ trạng thái có thể bị thay đổi bởi một lần chạy thử rồi hoàn nguyên khi thoát.

    Gồm: bể AMM, giá oracle, người vay, bộ đếm mô hình, lợi nhuận thanh lý viên và trạng thái RNG
    (để việc dự báo KHÔNG làm lệch dãy số ngẫu nhiên của mô phỏng thật).
    """
    pools = {a: (p.reserve_token, p.reserve_usdc) for a, p in model.amm_pools.items()}
    oracle = dict(model.oracle_prices)
    ext = dict(model.external_target_prices)
    borrowers = [(b, dict(b.collaterals), b.debt, b.cash_usdc, b.health_factor,
                  b.is_liquidated, b.bad_debt_recorded) for b in model.borrowers()]
    counters = (model.total_liquidations, model.stalled_liquidations, model.bad_debt,
                len(model.liquidation_events), model.current_tick)
    liqs = [(l, l.total_profit, l.priority_fees_paid) for l in model.liquidators]
    rng_state = model._sim_rng.getstate()
    try:
        yield
    finally:
        for a, (rt, ru) in pools.items():
            model.amm_pools[a].reserve_token, model.amm_pools[a].reserve_usdc = rt, ru
        model.oracle_prices.clear(); model.oracle_prices.update(oracle)
        model.external_target_prices.clear(); model.external_target_prices.update(ext)
        for b, coll, debt, cash, hf, isl, bdr in borrowers:
            b.collaterals = coll
            b.debt, b.cash_usdc, b.health_factor = debt, cash, hf
            b.is_liquidated, b.bad_debt_recorded = isl, bdr
        (model.total_liquidations, model.stalled_liquidations, model.bad_debt,
         n_ev, model.current_tick) = counters
        del model.liquidation_events[n_ev:]
        for l, tp, pf in liqs:
            l.total_profit, l.priority_fees_paid = tp, pf
        model._sim_rng.setstate(rng_state)


@dataclass
class Rollout:
    trough: Dict[str, float]         # giá oracle thấp nhất từng tài sản trong cả cửa sổ
    bonus_paid: float                # thưởng thanh lý người vay mất trong cửa sổ
    bad_debt_delta: float            # nợ xấu phát sinh thêm
    n_liquidations: int


def rollout(model: DeFiContagionModel, repay: Dict[int, float], horizon: int) -> Rollout:
    """Chạy thử `horizon` tick với các khoản trả nợ `repay` (id -> USDC) áp ngay tại tick hiện tại.

    Tick hiện tại: (đã qua bước người vay) -> Liquidator -> Arbitrageur. Các tick sau: đủ 3 bước theo
    thứ tự thật của model.step. Người chơi không hành động thêm trong cửa sổ (chỉ lập kế hoạch một lần).
    Mọi thay đổi trạng thái bị hoàn nguyên khi kết thúc.
    """
    by_id = {b.unique_id: b for b in model.borrowers()}
    with sandbox(model):
        n_ev0, bd0 = len(model.liquidation_events), model.bad_debt
        for uid, amt in repay.items():
            b = by_id[uid]
            b.debt -= amt
            b.cash_usdc -= amt
            b.health_factor = b.calculate_health_factor()
        trough = dict(model.oracle_prices)
        for h in range(horizon):
            if h > 0:
                for b in model.borrowers():
                    b.step()
            model.liquidator.step()
            model.arbitrageur.step()
            for a in ASSETS:
                trough[a] = min(trough[a], model.oracle_prices[a])
        events = model.liquidation_events[n_ev0:]
        bonus = sum(e["debt_covered"] * model.liquidation_bonuses[e["seized_asset"]] for e in events)
        return Rollout(trough, bonus, model.bad_debt - bd0, len(events))


# --------------------------------------------------------------------------- lợi ích người chơi
def predicted_hf(model: DeFiContagionModel, b: BorrowerAgent, prices: Dict[str, float], x: float) -> float:
    """HF của b tại giá `prices` nếu trả trước tỷ lệ x của nợ hiện tại."""
    debt = b.debt * (1.0 - x)
    if debt <= 0:
        return 999.0
    lts = model.liquidation_thresholds
    return sum(q * prices[a] * lts[a] for a, q in b.collaterals.items()) / debt


def avg_bonus(model: DeFiContagionModel, b: BorrowerAgent) -> float:
    """Thưởng thanh lý bình quân theo tỷ trọng giá trị thế chấp (ước lượng tài sản bị tịch thu)."""
    vals = {a: q * model.oracle_prices[a] for a, q in b.collaterals.items() if q > 0}
    tot = sum(vals.values())
    if tot <= 0:
        return 0.0
    return sum(v * model.liquidation_bonuses[a] for a, v in vals.items()) / tot


def expected_cost(model: DeFiContagionModel, cfg: GameConfig, b: BorrowerAgent,
                  prices: Dict[str, float], x: float) -> float:
    """Chi phí kỳ vọng (USD) của người vay b khi chọn x, biết giá đáy `prices`."""
    hf = predicted_hf(model, b, prices, x)
    z = (hf - 1.0) / cfg.tau
    p_liq = 0.0 if z > 700 else 1.0 / (1.0 + math.exp(z))
    loss = model.close_factor * b.debt * (1.0 - x) * (avg_bonus(model, b) + cfg.liq_extra_penalty)
    return p_liq * loss + cfg.phi * x * b.debt


def best_response(model: DeFiContagionModel, cfg: GameConfig, b: BorrowerAgent,
                  prices: Dict[str, float]) -> float:
    """x tối thiểu hóa chi phí kỳ vọng trên lưới khả thi (hòa thì chọn x nhỏ hơn)."""
    cap = min(cfg.max_repay_frac, b.cash_usdc / b.debt) if b.debt > 0 else 0.0
    best_x, best_c = 0.0, expected_cost(model, cfg, b, prices, 0.0)
    for x in cfg.grid[1:]:
        if x > cap + 1e-12:
            break
        c = expected_cost(model, cfg, b, prices, x)
        if c < best_c - 1e-9:
            best_x, best_c = x, c
    return best_x


# --------------------------------------------------------------------------- trò chơi
class DeleveragingGame:
    """Giải và áp dụng trò chơi giảm đòn bẩy tại mỗi tick (gọi từ GameTheoryModel.step)."""

    def __init__(self, cfg: GameConfig) -> None:
        self.cfg = cfg
        self.log: List[Dict] = []

    # -- chọn người chơi
    def players(self, model: DeFiContagionModel) -> List[BorrowerAgent]:
        out = []
        for b in model.borrowers():
            if b.debt <= 0 or b.cash_usdc <= 0:
                continue
            if self.cfg.players == "active" and not getattr(b, "is_player", False):
                continue
            if not self.cfg.allow_cure and b.calculate_health_factor() < 1.0:
                continue
            out.append(b)
        return out

    def _repay_map(self, players: List[BorrowerAgent], prof: Dict[int, float]) -> Dict[int, float]:
        return {b.unique_id: prof[b.unique_id] * b.debt
                for b in players if prof.get(b.unique_id, 0.0) > 0}

    # -- Nash
    def _regret(self, model, players, prof, trough) -> float:
        """Hối tiếc lớn nhất (tỷ lệ nợ) của một người chơi khi đáp trả đáy giá `trough` thay vì chiến lược hiện tại."""
        eps = 0.0
        for b in players:
            x_br = best_response(model, self.cfg, b, trough)
            r = (expected_cost(model, self.cfg, b, trough, prof[b.unique_id])
                 - expected_cost(model, self.cfg, b, trough, x_br))
            eps = max(eps, r / b.debt)
        return eps

    def _solve_nash(self, model, players) -> Tuple[Dict[int, float], Dict]:
        """Fictitious play xuất phát từ niềm tin bi quan (không ai trả nợ => đáy giá của cú thác).

        Mỗi vòng: hồ sơ = đáp trả tốt nhất với niềm tin; chạy thử để lấy đáy giá THỰC của hồ sơ đó; hồ sơ là
        cân bằng Nash (trên lưới) nếu mọi người đã đáp trả tốt nhất với đáy giá thực (epsilon = 0). Nếu chưa,
        niềm tin dịch về đáy giá thực rồi lặp. Trả về hồ sơ có epsilon nhỏ nhất đã gặp.
        Trò chơi có thể có nhiều cân bằng (xem docstring đầu file); nghiệm trả về là cân bằng đạt được từ niềm tin bi quan.
        """
        cfg = self.cfg
        belief = rollout(model, {}, cfg.horizon).trough
        best_prof, best_eps, it = {b.unique_id: 0.0 for b in players}, math.inf, 0
        for it in range(1, cfg.max_iter + 1):
            prof = {b.unique_id: best_response(model, cfg, b, belief) for b in players}
            actual = rollout(model, self._repay_map(players, prof), cfg.horizon).trough
            eps = self._regret(model, players, prof, actual)
            if eps < best_eps - 1e-12:
                best_prof, best_eps = prof, eps
            if eps <= 1e-9:
                break
            belief = {a: belief[a] + (actual[a] - belief[a]) / (it + 1) for a in ASSETS}
        return best_prof, {"iterations": it, "converged": best_eps <= 1e-9, "epsilon": best_eps}

    # -- nhà hoạch định
    def _social_cost(self, model, players, prof) -> float:
        r = rollout(model, self._repay_map(players, prof), self.cfg.horizon)
        repaid = sum(prof[b.unique_id] * b.debt for b in players if b.unique_id in prof)
        return r.bonus_paid + r.bad_debt_delta + self.cfg.phi * repaid

    def _solve_planner(self, model, players) -> Tuple[Dict[int, float], Dict]:
        """Nhà hoạch định: (1) chọn x chung cho nhóm đang rủi ro, (2) tối ưu tọa độ từng người (planner_sweeps lượt).

        Là tối ưu CỤC BỘ trên lưới (mỗi đánh giá là một lần chạy thử), không đảm bảo tối ưu toàn cục.
        """
        cfg = self.cfg
        null = rollout(model, {}, cfg.horizon).trough
        at_risk = [b for b in players if predicted_hf(model, b, null, 0.0) < cfg.risk_band]
        zero = {b.unique_id: 0.0 for b in players}
        best, best_c = dict(zero), self._social_cost(model, players, zero)
        for x in cfg.grid[1:]:
            prof = dict(zero)
            prof.update({b.unique_id: min(x, b.cash_usdc / b.debt) for b in at_risk})
            c = self._social_cost(model, players, prof)
            if c < best_c - 1e-9:
                best, best_c = prof, c
        evals = len(cfg.grid)
        for _ in range(cfg.planner_sweeps if best_c > 0 else 0):
            changed = False
            for b in sorted(at_risk, key=lambda q: -q.debt):
                cap = min(cfg.max_repay_frac, b.cash_usdc / b.debt)
                for x in cfg.grid:
                    if x > cap + 1e-12 or x == best[b.unique_id]:
                        continue
                    trial = dict(best)
                    trial[b.unique_id] = x
                    c = self._social_cost(model, players, trial)
                    evals += 1
                    if c < best_c - 1e-9:
                        best, best_c, changed = trial, c, True
            if not changed:
                break
        return best, {"iterations": evals, "converged": True, "epsilon": 0.0}

    # -- áp dụng
    def play(self, model: DeFiContagionModel) -> None:
        players = self.players(model)
        if not players:
            return
        solve = self._solve_nash if self.cfg.mode == "nash" else self._solve_planner
        prof, info = solve(model, players)
        total = 0.0
        n_rep = 0
        for b in players:
            x = prof.get(b.unique_id, 0.0)
            if x <= 0:
                continue
            amt = min(x * b.debt, b.cash_usdc)
            b.debt -= amt
            b.cash_usdc -= amt
            b.health_factor = b.calculate_health_factor()
            total += amt
            n_rep += 1
        self.log.append({"tick": model.current_tick, "n_players": len(players),
                         "n_repaying": n_rep, "repaid_usd": total, **info})


# --------------------------------------------------------------------------- mô hình tích hợp
class GameTheoryModel(DeFiContagionModel):
    """DeFiContagionModel + module trò chơi giảm đòn bẩy. `game=None` cho đúng kết quả mô hình gốc."""

    def __init__(self, *args, game: Optional[GameConfig] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.game_cfg = game
        self.game: Optional[DeleveragingGame] = DeleveragingGame(game) if game else None
        if game is not None:
            for b in self.borrowers():
                was_active = b.risk_profile == "active"
                b.is_player = was_active or game.players == "all"
                if b.is_player:
                    b.risk_profile = "strategic"   # tắt luật cố định cũ, thay bằng trò chơi

    def step(self) -> None:
        """Như model.step, thêm bước chơi trò chơi sau khi người vay cập nhật HF, trước Liquidator."""
        self.current_tick += 1
        if self.current_tick == 1:
            self.inject_shock()
        for b in self._borrowers:
            b.step()
        if self.game is not None and self.shock_applied:
            self.game.play(self)
        self.liquidator.step()
        self.arbitrageur.step()
        for b in self._borrowers:
            b.health_factor = b.calculate_health_factor()
        self.datacollector.collect(self)
