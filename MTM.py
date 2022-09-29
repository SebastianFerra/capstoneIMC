from Calculator import Calculator as Calc
from datetime import datetime as dt
import Factory as Fct
import pickle
import pandas as pd
## hacer cambios para que getinputs pueda usar data.csv y se vea como curva_inicial 
def get_inputs(market_date,paths):
    """
    order fot paths:

    0- fx_rates
    1- curves
    2-fixings
    3-dict_fixing_curves
    4-derivatives
    ex= ['inputs/tipos_de_cambio_inicial20220913.csv',  # sacar info de vol.csv uf(clf) spot y tc(usd) spot
        'inputs/curvainicial20220913.csv', # se trabajan cuando se cargan asi que afuera 
        'inputs/fixings_historicos.csv',
        'inputs/diccionario_fixings_curvas.csv',
        'inputs/pampavfusd.csv']
    """
    inputs_csv_delimiter = ';'
    inputs_decimal_separator = ','

    print('Cargando data de mercado...')
    fx_rates = pd.read_csv(paths[0], delimiter=inputs_csv_delimiter,
                           decimal=inputs_decimal_separator)
    curves = pd.read_csv(paths[1], delimiter=inputs_csv_delimiter, decimal=inputs_decimal_separator)
    fixings = pd.read_csv(paths[2], delimiter=inputs_csv_delimiter,
                          decimal=inputs_decimal_separator)
    dict_fixings_curves = pd.read_csv(paths[3], delimiter=inputs_csv_delimiter,
                                      decimal=inputs_decimal_separator)
    print('Formateando mercado...')
    market = Fct.get_market(market_date, fx_rates, curves, dict_fixings_curves, fixings)

    print('Cargando data de cartera...')
    derivs_df = pd.read_csv(paths[4], delimiter=inputs_csv_delimiter,
                            decimal=inputs_decimal_separator, encoding='cp1252')

    print('Formateando cartera...')
    portfolio = Fct.get_derivatives(derivs_df)
    return market, portfolio

def valorize_mtm(date,paths):
    """
    date in format: yyyy, mm, dd
    paths: list with paths to use in "get_inputs"
    """
    mkt_date = dt(date[0], date[1], date[2])
    market_or, portfolio = get_inputs(mkt_date, paths=paths)
    ops, mtm_op = zip(*[(d.operation_number, Calc.get_mtm(d, market_or, 'clp', print_errors=True)) for d in list(portfolio.values())])
    pd.DataFrame(zip(ops, mtm_op), columns=['Operacion', 'MtM Python']).to_csv('MtM_Curvas_RF.csv', sep=';',
                                                                               decimal=',', index=False)
    return sum(mtm_op)


if __name__ == '__main__':
    date = [2022,9,13]
    market_date = dt(2022, 9, 13)
    ex= ['inputs/tipos_de_cambio_inicial20220913.csv',
        'inputs/curvainicial20220913.csv',
        'inputs/fixings_historicos.csv',
        'inputs/diccionario_fixings_curvas.csv',
        'inputs/pampavfusd.csv']
    print(valorize_mtm(date = date, paths = ex))