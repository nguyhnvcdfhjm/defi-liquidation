"""
agents.py - Các lớp tác nhân trong mô hình Agent-Based (Mesa 3.x).
"""
from typing import Dict, List, Tuple, Optional

import mesa


class BorrowerAgent(mesa.Agent):
    """Người vay trong giao thức cho vay thế chấp gộp."""

    def __init__(self, model: mesa.Model, collaterals: Dict[str, float], debt: float,
                 group_id: int, risk_profile: str = "passive") -> None:
        """
        Khởi tạo ví người vay với số dư thế chấp, nợ và tiền mặt dự phòng:
            cash_usdc = debt * 0.30 (ngân sách de-leveraging), health_factor = 1.5,
            bad_debt_recorded = 0.0.
        group_id: 0 = Nhóm 1 (WETH+WBTC), 1 = Nhóm 2 (WBTC+LINK), 2 = Nhóm 3 (LINK).
        """
        super().__init__(model)
        self.collaterals: Dict[str, float] = dict(collaterals)
        self.debt: float = float(debt)
        self.group_id: int = group_id
        self.risk_profile: str = risk_profile
        self.cash_usdc: float = self.debt * 0.30
        self.health_factor: float = 1.5
        self.bad_debt_recorded: float = 0.0
        self.is_liquidated: bool = False

    def market_value(self) -> float:
        """V_mkt = sum(amount_i * oracle_price_i) (không nhân LT)."""
        prices = self.model.oracle_prices
        return sum(a * prices[s] for s, a in self.collaterals.items())

    def calculate_health_factor(self) -> float:
        """HF = sum(amount_i * oracle_price_i * lt_i) / debt; nợ <= 0 -> 999.0."""
        if self.debt <= 0:
            return 999.0
        prices = self.model.oracle_prices
        lts = self.model.liquidation_thresholds
        v_coll = sum(a * prices[s] * lts[s] for s, a in self.collaterals.items())
        return v_coll / self.debt

    def step(self) -> None:
        """Cập nhật HF; người vay 'active' tự trả nợ (tối đa 20% nợ, giới hạn bởi tiền mặt) khi 1.0 < HF < 1.15."""
        self.health_factor = self.calculate_health_factor()
        if (self.risk_profile == "active" and 1.0 < self.health_factor < 1.15
                and self.cash_usdc > 0):
            repay = min(self.debt * 0.20, self.cash_usdc)
            self.debt -= repay
            self.cash_usdc -= repay
            self.health_factor = self.calculate_health_factor()


def english_auction(gross: float, costs: List[float]) -> Tuple[Optional[int], float, float]:
    """
    Đấu giá phí ưu tiên (PGA) mô hình hóa như đấu giá kiểu Anh (giá tăng dần).

    Bot i thu được tối đa `gross` (lợi nhuận gộp của khoản thanh lý, chưa trừ gas) và phải trả
    chi phí gas riêng costs[i], nên GIÁ TRỊ của nó là v_i = gross - costs[i].
    Chiến lược trội (dominant strategy): mỗi bot tăng phí ưu tiên cho tới khi bằng v_i của mình.
    Kết quả cân bằng:
        - Người thắng  : bot có v_i lớn nhất (tức chi phí gas thấp nhất).
        - Phí ưu tiên  : bằng giá trị cao thứ hai, v_(2) = gross - costs_(2)   (nếu v_(2) > 0).
        - Lợi nhuận của người thắng = v_(1) - v_(2) = costs_(2) - costs_(1).
    Nếu chỉ có 1 bot (không cạnh tranh): phí ưu tiên = 0, lợi nhuận = gross - costs[0].
    Nếu không bot nào có v_i > 0: không ai thanh lý (trả về người thắng = None).

    Returns: (chỉ số người thắng hoặc None, phí ưu tiên trả cho thợ đào, lợi nhuận ròng người thắng).
    """
    values = [gross - c for c in costs]
    winner = max(range(len(values)), key=values.__getitem__)
    if values[winner] <= 0:
        return None, 0.0, 0.0
    if len(values) == 1:
        return winner, 0.0, values[winner]
    second_value = sorted(values)[-2]
    priority_fee = max(0.0, second_value)
    return winner, priority_fee, values[winner] - priority_fee


class LiquidatorAgent(mesa.Agent):
    """Bot thanh lý tối đa hóa lợi nhuận ròng."""

    def __init__(self, model: mesa.Model, gas_cost: float = 20.0) -> None:
        super().__init__(model)
        self.gas_cost: float = gas_cost
        self.total_profit: float = 0.0
        self.priority_fees_paid: float = 0.0  # tổng phí ưu tiên đã trả khi thắng đấu giá gas

    def _plan_gross(self, borrower: BorrowerAgent, asset: str,
                    desired_debt: float) -> Optional[Tuple[float, float, float]]:
        """
        Với tài sản `asset`: tính (a_actual, d_actual, lợi nhuận GỘP) theo quy tắc giới hạn thế chấp.
            a_req = d_desired * P_debt / P_k * (1 + beta)
            nếu a_req > số dư: a_actual = số dư, d_actual = a_actual * P_k / (P_debt * (1 + beta))
            lợi nhuận gộp = get_amount_out(a_actual) - d_actual * P_debt      (CHƯA trừ gas)
        """
        model = self.model
        pool = model.amm_pools.get(asset)
        available = borrower.collaterals.get(asset, 0.0)
        if pool is None or available <= 0 or desired_debt <= 0:
            return None
        p_debt = model.oracle_prices["USDC"]
        p_k = model.oracle_prices[asset]
        bonus = model.liquidation_bonuses[asset]
        required = desired_debt * p_debt * (1.0 + bonus) / p_k
        if available < required:
            actual_seize = available
            actual_debt = actual_seize * p_k / (p_debt * (1.0 + bonus))
        else:
            actual_seize = required
            actual_debt = desired_debt
        gross = pool.get_amount_out(actual_seize) - actual_debt * p_debt
        return actual_seize, actual_debt, gross

    def _plan(self, borrower: BorrowerAgent, asset: str,
              desired_debt: float) -> Optional[Tuple[float, float, float]]:
        """Như _plan_gross nhưng trả về lợi nhuận Π = gộp - gas của CHÍNH bot này."""
        plan = self._plan_gross(borrower, asset, desired_debt)
        if plan is None:
            return None
        seize, debt, gross = plan
        return seize, debt, gross - self.gas_cost

    def _draw_costs(self) -> List[float]:
        """
        Chi phí gas riêng của từng bot cho khoản thanh lý này.
        1 bot: dùng đúng gas_cost (không ngẫu nhiên, giữ nguyên hành vi bản gốc).
        Nhiều bot: gas_cost * U(0.5, 1.5) cho mỗi bot (độ trễ, vị trí trong khối, vốn khác nhau).
        """
        bots = self.model.liquidators
        if len(bots) == 1:
            return [self.gas_cost]
        rng = self.model._sim_rng
        return [b.gas_cost * rng.uniform(0.5, 1.5) for b in bots]

    def step(self) -> None:
        """
        Chỉ bot đầu tiên (model.liquidators[0]) chạy vòng thanh lý của cả thị trường; các bot còn lại
        không có hành động riêng mà tham gia đấu giá gas bên trong vòng này.
        1. Quét borrower có HF < 1.0 và debt > 0, nợ lớn nhất trước.
        2. Với mỗi người, chọn tài sản k* cho lợi nhuận cao nhất.
        3. Các bot đấu giá phí ưu tiên (english_auction). Có người thắng (giá trị > 0): người đó swap
           trên AMM, giảm nợ/thế chấp, oracle cập nhật theo giá AMM mới, ghi log.
        4. Không ai có lời: nếu V_mkt < nợ thì ghi nhận nợ xấu tăng thêm (không đếm trùng).
        """
        model = self.model
        targets: List[BorrowerAgent] = [
            b for b in model.borrowers() if b.health_factor < 1.0 and b.debt > 0]
        targets.sort(key=lambda b: b.debt, reverse=True)

        for borrower in targets:
            # Giá có thể đã đổi do thanh lý trước đó trong cùng tick.
            borrower.health_factor = borrower.calculate_health_factor()
            if borrower.health_factor >= 1.0 or borrower.debt <= 0:
                continue
            desired_debt = borrower.debt * model.close_factor

            best_asset, best_plan = None, None
            for asset in borrower.collaterals:
                plan = self._plan(borrower, asset, desired_debt)
                if plan is not None and (best_plan is None or plan[2] > best_plan[2]):
                    best_asset, best_plan = asset, plan

            winner_idx, priority_fee, costs = None, 0.0, []
            if best_plan is not None:
                seize, debt_cov, _ = best_plan
                gross = self._plan_gross(borrower, best_asset, desired_debt)[2]
                costs = self._draw_costs()
                winner_idx, priority_fee, _ = english_auction(gross, costs)

            if winner_idx is not None:
                winner = model.liquidators[winner_idx]
                pool = model.amm_pools[best_asset]
                usdc_out, new_spot = pool.swap_token_for_usdc(seize)
                profit = (usdc_out - debt_cov * model.oracle_prices["USDC"]
                          - costs[winner_idx] - priority_fee)

                left = borrower.collaterals[best_asset] - seize
                borrower.collaterals[best_asset] = left if left > 1e-12 else 0.0
                borrower.debt = max(0.0, borrower.debt - debt_cov)
                borrower.is_liquidated = True
                borrower.health_factor = borrower.calculate_health_factor()

                model.sync_oracle(best_asset, new_spot)  # oracle cập nhật theo giá AMM (xem model.sync_oracle)
                winner.total_profit += profit
                winner.priority_fees_paid += priority_fee
                model.total_liquidations += 1
                model.liquidation_events.append({
                    "tick": model.current_tick,
                    "borrower_id": borrower.unique_id,
                    "group_id": borrower.group_id,
                    "seized_asset": best_asset,
                    "seized_amount": seize,
                    "debt_covered": debt_cov,
                    "profit": profit,
                    "new_spot_price": new_spot,
                    # --- trường bổ sung cho phần cạnh tranh gas ---
                    "liquidator_id": winner.unique_id,
                    "priority_fee": priority_fee,
                    "n_bidders": len(costs),
                })
            else:
                model.stalled_liquidations += 1
                v_mkt = borrower.market_value()
                if v_mkt < borrower.debt:
                    shortfall = borrower.debt - v_mkt
                    delta = max(0.0, shortfall - borrower.bad_debt_recorded)
                    model.bad_debt += delta
                    borrower.bad_debt_recorded = shortfall


class ArbitrageurAgent(mesa.Agent):
    """Bot cân bằng giá: kéo giá AMM về giá mục tiêu ngoại sinh (không bao giờ về giá trước sốc)."""

    def __init__(self, model: mesa.Model, recovery_speed: float = 0.08) -> None:
        super().__init__(model)
        self.recovery_speed: float = recovery_speed

    def step(self) -> None:
        model = self.model
        for asset, pool in model.amm_pools.items():
            pool.replenish_liquidity(model.external_target_prices[asset], self.recovery_speed)
            model.sync_oracle(asset, pool.spot_price)  # Oracle quan sát giá AMM
