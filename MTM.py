from Calculator import Calculator as Calc
from datetime import datetime as dt
import Factory as Fct
import pickle
import pandas as pd

def get_inputs(market_date):
    inputs_csv_delimiter = ';'
    inputs_decimal_separator = ','

    print('Cargando data de mercado...')
    fx_rates = pd.read_csv('inputs/tipos_de_cambio_inicial20220913.csv', delimiter=inputs_csv_delimiter,
                           decimal=inputs_decimal_separator)
    curves = pd.read_csv('inputs/curvainicial20220913.csv', delimiter=inputs_csv_delimiter, decimal=inputs_decimal_separator)
    fixings = pd.read_csv('inputs/fixings_historicos.csv', delimiter=inputs_csv_delimiter,
                          decimal=inputs_decimal_separator)
    dict_fixings_curves = pd.read_csv('inputs/diccionario_fixings_curvas.csv', delimiter=inputs_csv_delimiter,
                                      decimal=inputs_decimal_separator)
    print('Formateando mercado...')
    market = Fct.get_market(market_date, fx_rates, curves, dict_fixings_curves, fixings)

    print('Cargando data de cartera...')
    derivs_df = pd.read_csv('inputs/PRUEBA.csv', delimiter=inputs_csv_delimiter,
                            decimal=inputs_decimal_separator, encoding='cp1252')

    print('Formateando cartera...')
    portfolio = Fct.get_derivatives(derivs_df)
    return market, portfolio

if __name__ == '__main__':
    market_date = dt(2022, 9, 13)
    print('Obteniendo portfolio y mercado completo...')
    market_or, portfolio = get_inputs(market_date)
    ops, mtm_op = zip(*[(d.operation_number, Calc.get_mtm(d, market_or, 'clp', print_errors=True)) for d in list(portfolio.values())]) #antes corchete if d.operation_number == 1823
    pd.DataFrame(zip(ops, mtm_op), columns=['Operacion', 'MtM Python']).to_csv('MtM_Curvas_RF.csv', sep=';',
                                                                               decimal=',', index=False)
    MTM = sum(mtm_op)

    print("{:0,.0f}".format(MTM))