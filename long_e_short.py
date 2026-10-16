class LongEShort:
    """Estratégia long e short por cointegração: A*x + B*y = 0 +/- tolerância."""

    def calcular(self, preco_a, preco_b, coef_a, coef_b, tolerancia):
        """
        preco_a, preco_b: valores atuais dos ativos A e B (x e y).
        coef_a, coef_b: coeficientes da relação de cointegração (A e B).
        tolerancia: faixa aceitável em torno de zero.

        Retorna {'A': qtd, 'B': qtd}:
          positivo = long (comprar), negativo = short (vender), 0 = não operar.
        """
        spread = coef_a * preco_a + coef_b * preco_b

        # Dentro da tolerância: não precisa operar
        if abs(spread) <= tolerancia:
            return {'A': 0, 'B': 0}

        # Spread acima: A está caro em relação a B -> short em A e long em B
        if spread > 0:
            return {'A': -abs(coef_a), 'B': abs(coef_b)}

        # Spread abaixo: A está barato em relação a B -> long em A e short em B
        return {'A': abs(coef_a), 'B': -abs(coef_b)}


if __name__ == "__main__":
    estrategia = LongEShort()
    print(estrategia.calcular(preco_a=100, preco_b=48, coef_a=1, coef_b=-2, tolerancia=1))