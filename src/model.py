"""
model.py - Lớp điều phối trung tâm DeFiContagionModel và DataCollector.
"""
import inspect
import random
from typing import Dict, Any, List, Optional

import mesa

from config import MarketConfig
from amm import AMMPool
from agents import BorrowerAgent, LiquidatorAgent, ArbitrageurAgent

# --- Tham số quy mô quần thể người vay (đặc tả không quy định; đã hiệu chỉnh cho TEST-03/04) ---
MIN_COLLATERAL_USD = 25_000.0
MAX_COLLATERAL_USD = 75_000.0
SEED_HF_RANGE = (1.18, 1.35)   # HF ban đầu >= 1.18
ACTIVE_BORROWER_RATIO = 0.3

# group_id (0-based) -> tài sản thế chấp
GROUP_ASSETS = {
    0: ("WETH", "WBTC"),   # Nhóm 1
    1: ("WBTC", "LINK"),   # Nhóm 2 - cầu nối lây chéo
    2: ("LINK",),          # Nhóm 3
}


class DeFiContagionModel(mesa.Model):
    """Điều phối không gian tác nhân, thời gian và sóng thanh lý chéo."""

    def __init__(
        self,
        n_borrowers: int = 60,
        shock_asset: str = "WETH",
        shock_pct: float = 0.35,
        amm_depth_multiplier: float = 1.0,
        close_factor: float = 0.50,
        seed: Optional[int] = 42,
        oracle_mode: str = "amm",
        n_liquidators: int = 1,
        liquidation_thresholds: Optional[Dict[str, float]] = None,
        liquidation_bonuses: Optional[Dict[str, float]] = None,
        shock_pcts: Optional[Dict[str, float]] = None,
    ) -> None:
        # 1. Seed (tương thích Mesa 3.0 -> 3.5+: tham số `seed` hay `rng`).
        params = inspect.signature(mesa.Model.__init__).parameters
        if "rng" in params:
            super().__init__(rng=seed)
        else:
            super().__init__(seed=seed)
        # Bộ sinh ngẫu nhiên riêng cho việc dựng quần thể -> tái lập độc lập phiên bản Mesa.
        self._sim_rng = random.Random(seed)

        cfg = MarketConfig()
        if shock_asset not in cfg.amm_base_reserves:
            raise ValueError(f"shock_asset phải thuộc {list(cfg.amm_base_reserves)}")
        if not 0.0 < shock_pct < 1.0:
            raise ValueError("shock_pct phải thuộc (0, 1)")
        if not 0.0 < close_factor <= 1.0:
            raise ValueError("close_factor phải thuộc (0, 1]")
        if amm_depth_multiplier <= 0:
            raise ValueError("amm_depth_multiplier phải > 0")
        if n_liquidators < 1:
            raise ValueError("n_liquidators phải >= 1")
        if oracle_mode not in ("amm", "external"):
            raise ValueError("oracle_mode phải là 'amm' hoặc 'external'")

        self.config = cfg
        # shock_pcts (tuỳ chọn): sốc ĐA tài sản, {tài sản: mức giảm}, áp đồng thời tại tick 1.
        # None = sốc đơn như cũ: {shock_asset: shock_pct}. Khi đa tài sản, shock_asset/shock_pct
        # trỏ tới tài sản bị sốc nặng nhất (chỉ để các hàm báo cáo cũ vẫn chạy).
        if shock_pcts is not None:
            if not shock_pcts:
                raise ValueError("shock_pcts không được rỗng")
            for a, pct in shock_pcts.items():
                if a not in cfg.amm_base_reserves:
                    raise ValueError(f"shock_pcts: tài sản {a} không hợp lệ")
                if not 0.0 < pct < 1.0:
                    raise ValueError("mỗi mức sốc trong shock_pcts phải thuộc (0, 1)")
            self.shocks: Dict[str, float] = dict(shock_pcts)
            shock_asset = max(self.shocks, key=self.shocks.get)
            shock_pct = self.shocks[shock_asset]
        else:
            self.shocks = {shock_asset: shock_pct}
        self.shock_asset = shock_asset
        self.shock_pct = shock_pct
        self.amm_depth_multiplier = amm_depth_multiplier
        self.close_factor = close_factor
        # oracle_mode = "amm"      : oracle bám giá AMM  -> kênh lây lan BẬT (đúng đặc tả)
        # oracle_mode = "external" : oracle chỉ theo giá ngoại sinh -> kênh lây lan TẮT (nhóm đối chứng)
        self.oracle_mode = oracle_mode
        self.sim_seed = seed  # tránh trùng tên thuộc tính nội bộ `seed` của Mesa

        # 2. Ba tầng giá.
        self.baseline_prices: Dict[str, float] = dict(cfg.baseline_prices)
        self.oracle_prices: Dict[str, float] = dict(cfg.baseline_prices)
        self.external_target_prices: Dict[str, float] = dict(cfg.baseline_prices)
        self.liquidation_thresholds: Dict[str, float] = dict(cfg.liquidation_thresholds)
        self.liquidation_bonuses: Dict[str, float] = dict(cfg.liquidation_bonuses)
        # Ghi đè tuỳ chọn (mặc định None = giữ nguyên cấu hình gốc). LT phải được đặt trước khi dựng quần thể.
        if liquidation_thresholds:
            self.liquidation_thresholds.update(liquidation_thresholds)
        if liquidation_bonuses:
            self.liquidation_bonuses.update(liquidation_bonuses)

        # 3. Bể AMM: hệ số độ sâu nhân vào CẢ HAI phía dự trữ (giữ nguyên giá ban đầu).
        self.amm_pools: Dict[str, AMMPool] = {
            token: AMMPool(token, r_tok * amm_depth_multiplier, r_usdc * amm_depth_multiplier,
                           cfg.amm_fee_rate)
            for token, (r_tok, r_usdc) in cfg.amm_base_reserves.items()
        }

        self.current_tick: int = 0
        self.shock_applied: bool = False
        self.total_liquidations: int = 0
        self.stalled_liquidations: int = 0
        self.bad_debt: float = 0.0
        self.liquidation_events: List[Dict[str, Any]] = []

        # 4. Quần thể người vay.
        self._borrowers: List[BorrowerAgent] = []
        self._seed_borrowers(n_borrowers)

        # 5. Liquidator và Arbitrageur.
        # n_liquidators = 1: một bot như bản gốc. > 1: các bot cạnh tranh gas (xem english_auction).
        self.liquidators = [LiquidatorAgent(self, gas_cost=cfg.default_gas_cost_usd)
                            for _ in range(n_liquidators)]
        self.liquidator = self.liquidators[0]  # bot điều phối vòng thanh lý của cả thị trường
        self.arbitrageur = ArbitrageurAgent(self, recovery_speed=cfg.default_arbitrage_recovery_rate)

        # 6. DataCollector (hàng đầu = trạng thái tick 0).
        self.datacollector = mesa.DataCollector(model_reporters={
            "Price_WETH": lambda m: m.oracle_prices["WETH"],
            "Price_WBTC": lambda m: m.oracle_prices["WBTC"],
            "Price_LINK": lambda m: m.oracle_prices["LINK"],
            "Spot_WETH": lambda m: m.amm_spot_prices["WETH"],
            "Spot_WBTC": lambda m: m.amm_spot_prices["WBTC"],
            "Spot_LINK": lambda m: m.amm_spot_prices["LINK"],
            "Total_Liquidations": "total_liquidations",
            "Bad_Debt": "bad_debt",
            "Insolvent_Count": lambda m: sum(
                1 for a in m.agents
                if isinstance(a, BorrowerAgent) and a.health_factor < 1.0 and a.debt > 0),
            "Avg_HF": lambda m: sum(
                a.health_factor for a in m.agents
                if isinstance(a, BorrowerAgent) and a.debt > 0
            ) / max(1, sum(1 for a in m.agents if isinstance(a, BorrowerAgent) and a.debt > 0)),
        })
        self.datacollector.collect(self)

    # ------------------------------------------------------------------ thuộc tính
    @property
    def amm_spot_prices(self) -> Dict[str, float]:
        """Giá giao ngay tức thời trên các bể AMM (chỉ đọc)."""
        return {t: p.spot_price for t, p in self.amm_pools.items()}

    def sync_oracle(self, asset: str, spot_price: float) -> None:
        """Cập nhật oracle của `asset`. Đây là điểm DUY NHẤT nối giá AMM với giá dùng tính Health Factor."""
        if self.oracle_mode == "amm":
            self.oracle_prices[asset] = spot_price
        else:
            self.oracle_prices[asset] = self.external_target_prices[asset]

    def borrowers(self) -> List[BorrowerAgent]:
        return self._borrowers

    # ------------------------------------------------------------------ dựng quần thể
    def _seed_borrowers(self, n_borrowers: int) -> None:
        n0 = round(0.4 * n_borrowers)
        n2 = round(0.2 * n_borrowers)
        n1 = max(0, n_borrowers - n0 - n2)
        rng = self._sim_rng
        for group_id, count in ((0, n0), (1, n1), (2, n2)):
            assets = GROUP_ASSETS[group_id]
            for _ in range(count):
                total_value = rng.uniform(MIN_COLLATERAL_USD, MAX_COLLATERAL_USD)
                if len(assets) == 1:
                    shares = [1.0]
                else:
                    w = rng.uniform(0.35, 0.65)
                    shares = [w, 1.0 - w]
                collaterals = {a: total_value * s / self.oracle_prices[a]
                               for a, s in zip(assets, shares)}
                weighted = sum(total_value * s * self.liquidation_thresholds[a]
                               for a, s in zip(assets, shares))
                hf0 = rng.uniform(*SEED_HF_RANGE)
                profile = "active" if rng.random() < ACTIVE_BORROWER_RATIO else "passive"
                b = BorrowerAgent(self, collaterals, debt=weighted / hf0,
                                  group_id=group_id, risk_profile=profile)
                b.health_factor = b.calculate_health_factor()
                self._borrowers.append(b)

    # ------------------------------------------------------------------ động lực học
    def inject_shock(self) -> None:
        """
        Cú sốc ngoại sinh (áp dụng một lần): giá mới = baseline * (1 - shock_pct) cho oracle và
        giá mục tiêu ngoại sinh; bể AMM của tài sản sốc được điều chỉnh về giá đó (k giữ nguyên).
        """
        if self.shock_applied:
            return
        for a, pct in self.shocks.items():   # mọi tài sản bị sốc trong CÙNG một tick
            new_price = self.baseline_prices[a] * (1.0 - pct)
            self.oracle_prices[a] = new_price
            self.external_target_prices[a] = new_price
            self.amm_pools[a].replenish_liquidity(new_price, recovery_rate=1.0)
        self.shock_applied = True

    def step(self) -> None:
        """Thứ tự nghiêm ngặt: sốc (tick 1) -> Borrowers -> Liquidator -> Arbitrageur -> DataCollector."""
        self.current_tick += 1
        if self.current_tick == 1:
            self.inject_shock()
        for b in self._borrowers:
            b.step()
        self.liquidator.step()
        self.arbitrageur.step()
        for b in self._borrowers:  # làm mới HF theo giá cuối tick cho số liệu thu thập
            b.health_factor = b.calculate_health_factor()
        self.datacollector.collect(self)
