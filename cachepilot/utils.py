"""CachePilot 各子系统共享的无状态基础工具。"""

from __future__ import annotations

from typing import Type


class CommonUtils:
    """集中提供数值换算和基础参数校验的静态工具方法。

    校验方法允许调用方传入自己的异常类型，因此 Admission、Scheduler、
    Executor 等模块可以共享实现，同时保持各自稳定的公共异常契约。
    """

    @staticmethod
    def require_identifier(
        value: str,  # 要校验的标识符值。
        field_name: str,  # 错误消息中使用的字段名称。
        error_type: Type[Exception] = ValueError,  # 校验失败时抛出的异常类型。
    ) -> str:
        """要求值为非空字符串，并返回已校验值。"""

        if type(value) is not str or not value:
            raise error_type(
                "{0} must be a non-empty string".format(field_name)
            )
        return value

    @staticmethod
    def require_positive_int(
        value: int,  # 要校验的整数值。
        field_name: str,  # 错误消息中使用的字段名称。
        error_type: Type[Exception] = ValueError,  # 校验失败时抛出的异常类型。
    ) -> int:
        """要求值为正整数并显式拒绝 bool，返回已校验值。"""

        if type(value) is not int or value < 1:
            raise error_type(
                "{0} must be a positive integer".format(field_name)
            )
        return value

    @staticmethod
    def require_non_negative_int(
        value: int,  # 要校验的整数值。
        field_name: str,  # 错误消息中使用的字段名称。
        error_type: Type[Exception] = ValueError,  # 校验失败时抛出的异常类型。
    ) -> int:
        """要求值为非负整数并显式拒绝 bool，返回已校验值。"""

        if type(value) is not int or value < 0:
            raise error_type(
                "{0} must be a non-negative integer".format(field_name)
            )
        return value

    @staticmethod
    def ceil_div(
        dividend: int,  # 非负被除数。
        divisor: int,  # 正整数除数。
    ) -> int:
        """对非负整数执行无浮点向上整除。"""

        CommonUtils.require_non_negative_int(dividend, "dividend")
        CommonUtils.require_positive_int(divisor, "divisor")
        if dividend == 0:
            return 0
        return (dividend + divisor - 1) // divisor

    @staticmethod
    def milliseconds_to_ns(
        value: int,  # 要转换为纳秒的非负整数毫秒值。
    ) -> int:
        """使用整数运算把毫秒转换为纳秒。"""

        CommonUtils.require_non_negative_int(value, "milliseconds")
        return value * 1_000_000
