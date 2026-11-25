"""
config.py - Tập trung toàn bộ hằng số thị trường và tham số kịch bản mặc định.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class MarketConfig:
    """Tham số thị trường cơ sở."""
    baseline_prices: Dict[str, float] = field(default_factory=lambda: {
        'WETH': 3000.0,
        'WBTC': 60000.0,
        'LINK': 15.0,
        'USDC': 1.0
    })
    liquidation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'WETH': 0.82,
        'WBTC': 0.75,
        'LINK': 0.65
    })
    liquidation_bonuses: Dict[str, float] = field(default_factory=lambda: {
        'WETH': 0.05,
        'WBTC': 0.08,
        'LINK': 0.12
    })
    amm_base_reserves: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        # Format: token_symbol -> (reserve_token, reserve_usdc)
        'WETH': (500.0, 1_500_000.0),
        'WBTC': (15.0, 900_000.0),
        'LINK': (50_000.0, 750_000.0)
    })
    amm_fee_rate: float = 0.003
    default_gas_cost_usd: float = 20.0
    default_close_factor: float = 0.50
    default_arbitrage_recovery_rate: float = 0.08
