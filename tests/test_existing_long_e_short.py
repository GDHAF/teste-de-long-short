"""Testes de preservação da funcionalidade existente em long_e_short.py."""

from long_e_short import LongEShort


def test_long_e_short_dentro_da_tolerancia():
    estrategia = LongEShort()
    # spread = 1*100 + (-2)*50 = 0 <= 1
    resultado = estrategia.calcular(
        preco_a=100, preco_b=50, coef_a=1, coef_b=-2, tolerancia=1
    )
    assert resultado == {"A": 0, "B": 0}


def test_long_e_short_spread_positivo_short_a_long_b():
    estrategia = LongEShort()
    # spread = 1*100 + (-2)*48 = 4 > 1
    resultado = estrategia.calcular(
        preco_a=100, preco_b=48, coef_a=1, coef_b=-2, tolerancia=1
    )
    assert resultado == {"A": -1, "B": 2}


def test_long_e_short_spread_negativo_long_a_short_b():
    estrategia = LongEShort()
    # spread = 1*95 + (-2)*50 = -5 < -1
    resultado = estrategia.calcular(
        preco_a=95, preco_b=50, coef_a=1, coef_b=-2, tolerancia=1
    )
    assert resultado == {"A": 1, "B": -2}
