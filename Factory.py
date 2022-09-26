from Calculator import Market as mkt, Derivatives as dvt
import pandas as pd


def get_market(market_date, fx_df, curves_df, fixing_dict_df, fixing_history_df):
    curves_df['codigocurva'] = curves_df['codigocurva'].str.lower()
    fixing_dict_df['NombreIndice'] = fixing_dict_df['NombreIndice'].str.lower()
    fixing_dict_df['NombreCurva'] = fixing_dict_df['NombreCurva'].str.lower()
    fixing_history_df['NombreIndice'] = fixing_history_df['NombreIndice'].str.lower()
    fixing_history_df['Fecha'] = pd.to_datetime(fixing_history_df['Fecha'], format='%d-%m-%Y')

    fixing_dict = dict(zip(fixing_dict_df.NombreIndice, fixing_dict_df.NombreCurva))
    fixing_history = {index_name: dict(zip(fixing_history_df[fixing_history_df['NombreIndice'] == index_name].Fecha,
                                                 fixing_history_df[fixing_history_df['NombreIndice'] == index_name].Valor))
                      for index_name in fixing_history_df['NombreIndice'].unique()}
    market_indexes = {index_name: mkt.Index(index_name, fixing_dict[index_name], fixing_history[index_name],
                                            is_overnight=True if index_name in ['icp_clp', 'icp_clf', 'fed', 'eonia', 'ibr_cop']
                                                              else False)
                      for index_name in fixing_history}
    fx_rates = dict(zip([mon.lower() for mon in fx_df.moneda], fx_df.valor))
    curves = {curve_name: dict(zip(
                                            curves_df[curves_df['codigocurva'] == curve_name].tenor,
                                            curves_df[curves_df['codigocurva'] == curve_name].DF))
              for curve_name in curves_df['codigocurva'].unique()}

    market = mkt.Market(market_indexes, curves, fx_rates, market_date)
    return market


def build_coupons(list_table_leg, col_idx_dict):
    """
    Transforma dataframe en cupones de derivado
    :param leg_df: DF_Leg: FechaFixing, FechaInicio, FechaVencimiento, FechaPago, NocionalRemanente, Amortizacion,
                                    TasaFija, ConvencionInteres, ConvencionDias, IndiceFlotante, Spread, TipoInteres
    :return:
    """
    coupons = []
    if list_table_leg[0][col_idx_dict['TipoInteres']].lower() == 'fijo' and list_table_leg[0][col_idx_dict['tipoproducto']].lower() == 'swap':
        coupons = [dvt.FixedCoupon(
                        row[col_idx_dict['NocionalRemanente']],
                        row[col_idx_dict['amortizacion']] + row[col_idx_dict['flujoadicional']],
                        {
                            'fixing_date': row[col_idx_dict['FechaFijacion']],
                            'start_date': row[col_idx_dict['FechaInicio']],
                            'end_date': row[col_idx_dict['FechaVencimiento']],
                            'payment_date': row[col_idx_dict['FechaPago']],
                         },
                        row[col_idx_dict['TasaFija']], row[col_idx_dict['ConvencionInteres']], row[col_idx_dict['ConvencionDias']])
                   for row in list_table_leg]
    elif list_table_leg[0][col_idx_dict['TipoInteres']].lower() == 'flotante':

        coupons = [dvt.FloatingCoupon(
            row[col_idx_dict['NocionalRemanente']],
            row[col_idx_dict['amortizacion']] + row[col_idx_dict['flujoadicional']],
            row[col_idx_dict['Spread']],
            {'fixing_date': row[col_idx_dict['FechaFijacion']],
             'start_date': row[col_idx_dict['FechaInicio']],
             'end_date': row[col_idx_dict['FechaVencimiento']],
             'payment_date': row[col_idx_dict['FechaPago']],
             },
            row[col_idx_dict['IndiceFlotante']], row[col_idx_dict['ConvencionInteres']], row[col_idx_dict['ConvencionDias']])
            for row in list_table_leg]
    elif list_table_leg[0][col_idx_dict['tipoproducto']].lower() == 'forward':
        coupons = [dvt.FXForwardCoupon(list_table_leg[0][col_idx_dict['NocionalRemanente']],
                                       list_table_leg[0][col_idx_dict['FechaPago']])]

    return coupons


def build_derivative(operation_number, raw_derivative, col_idx_dict):
    """
    Transforma diccionario (por pata) de dataframes en o objeto derivado de librería
    :param operation_number: int
    :param raw_derivative: {'derivative_type': string, 'collateral_currency': string,
                                    'active_leg': (DF_Leg, currency), 'pasivo_leg': (DF_Leg, currency)}:
                                    DF_Leg: FechaFixing, FechaFin, FechaPago, NocionalRemanente, Amortizacion,
                                    TasaFija, ConvencionInteres, ConvencionDias, IndiceFlotante, Spread
    :return:
    """
    derivative_type = raw_derivative['derivative_type']
    collateral_currency = raw_derivative['collateral_currency'].lower()
    active_leg_data = raw_derivative['active_leg']
    passive_leg_data = raw_derivative['passive_leg']
    active_leg = active_leg_data[0]
    passive_leg = passive_leg_data[0]
    active_leg_currency = active_leg_data[1].lower()
    passive_leg_currency = passive_leg_data[1].lower()

    active_coupons = build_coupons(active_leg, col_idx_dict)
    passive_coupons = build_coupons(passive_leg, col_idx_dict)
    active_leg = dvt.Leg(active_coupons, active_leg_currency)
    passive_leg = dvt.Leg(passive_coupons, passive_leg_currency)

    derivative = dvt.Derivative(operation_number, derivative_type, collateral_currency, active_leg, passive_leg)

    return derivative


def get_derivative_type(table, col_ix_dict):
    curr_num = len(set([row[col_ix_dict['moneda']] for row in table]))
    return 'Forward' if table[0][col_ix_dict['tipoproducto']].lower() == 'forward' else 'CCS' if curr_num == 2 else 'IRS'


def structure_raw_derivatives(raw_derivatives):
    column_names = list(raw_derivatives.head())
    derivatives_dict = raw_derivatives.groupby('numerooperacion')[column_names].apply(lambda g: g.values.tolist()).to_dict()

    col_idx_dict = {col_name: raw_derivatives.columns.get_loc(col_name) for col_name in column_names}

    structured_derivatives = {}
    for operation_number in derivatives_dict:
        list_table = derivatives_dict[operation_number]
        derivative_type = get_derivative_type(list_table, col_idx_dict)
        collateral_currency = list_table[0][col_idx_dict['MonedaColateral']].lower()
        active_table = [row for row in list_table if row[col_idx_dict['tipoflujo']] == 'ACT']
        active_currency = active_table[0][col_idx_dict['moneda']].lower()
        active_leg = (active_table, active_currency)
        passive_table = [row for row in list_table if row[col_idx_dict['tipoflujo']] == 'PAS']
        passive_currency = passive_table[0][col_idx_dict['moneda']].lower()
        passive_leg = (passive_table, passive_currency)
        structured_derivatives[operation_number] = {'derivative_type': derivative_type,
                                                    'collateral_currency': collateral_currency,
                                                    'active_leg': active_leg,
                                                    'passive_leg': passive_leg}

    return structured_derivatives, col_idx_dict


def get_derivatives(raw_derivatives):
    #raw_derivatives['amortizacion'] = raw_derivatives['amortizacion'].str.replace(',', '.')
    #raw_derivatives['NocionalRemanente'] = raw_derivatives['NocionalRemanente'].str.replace(',', '.')
    #raw_derivatives = raw_derivatives.astype({'amortizacion': 'float64', 'NocionalRemanente': 'float64'})
    raw_derivatives['FechaFijacion'] = pd.to_datetime(raw_derivatives['FechaFijacion'], format='%d-%m-%Y')
    raw_derivatives['FechaInicio'] = pd.to_datetime(raw_derivatives['FechaInicio'], format='%d-%m-%Y')
    raw_derivatives['FechaVencimiento'] = pd.to_datetime(raw_derivatives['FechaVencimiento'], format='%d-%m-%Y')
    raw_derivatives['FechaPago'] = pd.to_datetime(raw_derivatives['FechaPago'], format='%d-%m-%Y')
    raw_derivatives = raw_derivatives.sort_values(['numerooperacion', 'tipoflujo', 'FechaPago'])
    structured_derivatives_tup = structure_raw_derivatives(raw_derivatives)
    structured_derivatives = structured_derivatives_tup[0]
    col_idx_dict = structured_derivatives_tup[1]

    derivatives = {int(op_num): build_derivative(op_num, structured_derivatives[op_num], col_idx_dict)
                   for op_num in structured_derivatives}

    return derivatives

