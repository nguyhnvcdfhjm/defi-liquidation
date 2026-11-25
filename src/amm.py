"""
amm.py - Bể thanh khoản Constant Product DEX (x * y = k).
"""
import math
from typing import Tuple


class AMMPool:
    """
    Mô phỏng bể thanh khoản tự động chuẩn Uniswap v2 cho cặp Token / USDC.

    Quy tắc bất biến:
        (R_token + gamma * delta_x) * (R_usdc - delta_y) = R_token * R_usdc
    """

    def __init__(self, token_name: str, reserve_token: float, reserve_usdc: float,
                 fee_rate: float = 0.003) -> None:
        if reserve_token <= 0 or reserve_usdc <= 0:
            raise ValueError("Dự trữ ban đầu phải > 0")
        if not 0.0 <= fee_rate < 1.0:
            raise ValueError("fee_rate phải thuộc [0, 1)")
        self.token_name: str = token_name
        self.reserve_token: float = float(reserve_token)
        self.reserve_usdc: float = float(reserve_usdc)
        self.fee_rate: float = float(fee_rate)
        self.gamma: float = 1.0 - self.fee_rate
        self.initial_k: float = self.reserve_token * self.reserve_usdc

    @property
    def k(self) -> float:
        return self.reserve_token * self.reserve_usdc

    @property
    def spot_price(self) -> float:
        """Giá giao ngay: P = R_usdc / R_token."""
        if self.reserve_token <= 0:
            raise ValueError("reserve_token phải > 0")
        return self.reserve_usdc / self.reserve_token

    def get_amount_out(self, amount_in_token: float) -> float:
        """USDC nhận được khi bán amount_in_token, KHÔNG thay đổi trạng thái bể."""
        if amount_in_token < 0:
            raise ValueError("amount_in_token phải >= 0")
        if amount_in_token == 0:
            return 0.0
        effective_in = amount_in_token * self.gamma
        numerator = self.reserve_usdc * effective_in
        denominator = self.reserve_token + effective_in
        return numerator / denominator

    def swap_token_for_usdc(self, amount_in_token: float) -> Tuple[float, float]:
        """Hoán đổi thực tế. Trả về (usdc_out, giá giao ngay mới)."""
        if amount_in_token <= 0:
            raise ValueError("amount_in_token phải > 0")
        usdc_out = self.get_amount_out(amount_in_token)
        self.reserve_token += amount_in_token
        self.reserve_usdc -= usdc_out
        return usdc_out, self.spot_price

    def replenish_liquidity(self, target_price: float, recovery_rate: float = 0.08) -> None:
        """
        Arbitrageur kéo giá giao ngay về phía target_price: khoảng cách giá thu hẹp
        `recovery_rate` mỗi bước. Giữ nguyên tích k hiện tại khi dịch giá.
        """
        if not 0.0 < recovery_rate <= 1.0:
            raise ValueError("recovery_rate phải thuộc (0, 1]")
        if target_price <= 0:
            raise ValueError("target_price phải > 0")
        price = self.spot_price
        new_price = price + recovery_rate * (target_price - price)
        k = self.reserve_token * self.reserve_usdc
        self.reserve_token = math.sqrt(k / new_price)
        self.reserve_usdc = math.sqrt(k * new_price)
