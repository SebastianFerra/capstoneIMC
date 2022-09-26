import math
import numpy as np
from Calculator import Dates as D


def __extrapolate_df(curve, x, min_x, max_x, extrapolate_flat):
    df = 1
    if extrapolate_flat:
        ref_x = min_x if x < min_x else max_x
        rate = curve[ref_x]**(-360.0000/ref_x) - 1
        df = (1 + rate)**(-x/360.0000)
    else:
        raise NotImplementedError()

    return df


def __log_interpol(dict_param, pair_tenors, x):
    for previous_tenor, next_tenor in pair_tenors:
        if previous_tenor < x and x < next_tenor:
            y1_log = math.log(dict_param[previous_tenor])
            y2_log = math.log(dict_param[next_tenor])
            b = (y2_log - y1_log) / (next_tenor - previous_tenor)
            return math.exp(y1_log + b * (x - previous_tenor))

    raise Exception('log interpol error')


def complete_curve(curve, max_extrapolation_days, min_extrapolate_flat, max_extrapolate_flat):
    tenors = list(curve.keys())
    tenors.sort()
    pair_tenors = set(zip(tenors[:len(tenors)-1], tenors[1:]))
    min_tenor = int(min(tenors))
    max_tenor = int(max(tenors))

    curve = {**curve,
             **{tenor: __log_interpol(curve, pair_tenors, tenor)
                for tenor in range(min_tenor + 1, max_tenor) if tenor not in tenors}}

    curve = {**curve,
             **{tenor: __extrapolate_df(curve, tenor, min_tenor, max_tenor, min_extrapolate_flat)
                for tenor in range(0, min_tenor)}}

    curve = {**curve,
             **{tenor: __extrapolate_df(curve, tenor, min_tenor, max_tenor, max_extrapolate_flat)
                for tenor in range(max_tenor, max_extrapolation_days + 1)}}

    return curve


class Market:
    def __init__(self, indexes, df_curves, fx_rates, market_date):
        self.indexes = indexes if indexes is not None else {}
        self.__original_curves = df_curves
        self.curves = {}
        self.fx_rates = fx_rates
        self.__original_fx_rates = fx_rates
        if 'clp' not in fx_rates:
            fx_rates['clp'] = 1

        self.market_date = market_date
        self.__original_market_date = market_date
        self.__complete_curves()
        self.__add_collateral_adj_curve()

    def __copy__(self):
        return Market(self.indexes, self.__original_curves, self.fx_rates, self.__original_market_date)

    def get_fx(self, base_currency, value_currency):
        return self.fx_rates[base_currency.lower()] / self.fx_rates[value_currency.lower()]

    def age_curves(self, new_date):
        days_to_age = (new_date - self.market_date).days
        for curve_name, curve_dict in self.curves.items():
            for tenor, df in curve_dict.items():
                if tenor < days_to_age:
                    continue
                self.curves[curve_name][tenor - days_to_age] = df / curve_dict[days_to_age]

    def get_discount_dfs(self, currency, days):
        curve = self.curves['curva_' + currency.lower() + '_cl']
        return np.array([curve[day] for day in days])

    def get_original_curve(self, curve_name):
        return self.__original_curves[curve_name]

    def get_curve_names(self):
        return list(self.__original_curves.keys())

    def __complete_curves(self, max_extrapolation_days=40*365, min_extrapolate_flat=True, max_extrapolate_flat=True,
                          curve_name=None):
        if curve_name is None:
            for c_name in self.__original_curves:
                if (curve_name is not None and curve_name == c_name) or curve_name is None:
                    curve = self.__original_curves[c_name]
                    max_tenor = int(max(curve.keys()))
                    max_extrapolation_days_curve = min(2 * max_tenor, max_extrapolation_days)
                    self.curves[c_name] = complete_curve(curve, max_extrapolation_days_curve, min_extrapolate_flat, max_extrapolate_flat)
        else:
            curve = self.__original_curves[curve_name]
            max_tenor = int(max(curve.keys()))
            max_extrapolation_days_curve = min(2 * max_tenor, max_extrapolation_days)
            self.curves[curve_name] = complete_curve(curve, max_extrapolation_days_curve, min_extrapolate_flat,
                                                 max_extrapolate_flat)


    def set_original_curve(self, curve, curve_name):
        self.__original_curves[curve_name] = curve
        self.__complete_curves(curve_name=curve_name)
        if curve_name in ['curva_usd_cl', 'curva_usd_usa']:
            self.__add_collateral_adj_curve()

    def __add_collateral_adj_curve(self):
        self.curves['col_adj_curve'] = {tenor: self.curves['curva_usd_usa'][tenor] /
                                                        self.curves['curva_usd_cl'][tenor]
                                                 for tenor in self.curves['curva_usd_usa']}

    def shock_curve(self, bps, shock_type='Parallel', curve_name=None):
        if curve_name == None:
            curve_names = self.get_curve_names()
            for cn in curve_names:
                self.shock_curve(bps, shock_type, cn)
        else:
            if shock_type == 'Parallel':
                self.__parallel_shock(bps, curve_name)

    def __parallel_shock(self, bps, curve_name):
        t, fd = zip(*self.curves[curve_name].items())
        t = np.array(t)
        fd = np.array(fd)

        fd_shocked = (1 + (fd**(-360/t) - 1) + bps / 10000.000)**(-t/360)
        self.curves[curve_name] = dict(zip(t, fd_shocked))

    def recover_original_curves(self):
        self.__complete_curves()

    def recover_original_fx_rates(self):
        self.fx_rates = self.__original_fx_rates


class Index:
    def __init__(self, name, projection_curve_name, fixing, is_overnight=False):
        self.name = name
        self.projection_curve_name = projection_curve_name
        self.fixings = fixing
        self.is_overnight = is_overnight
        self.fixed_factors = self.__get_fixed_factors() if is_overnight else None

    def __get_fixed_factors(self):
        fixed_factors = {}
        fixing_dates = list(self.fixings.keys())
        fixing_dates.sort(reverse=True)
        max_fixing_date = max(fixing_dates)
        fixed_factors[max_fixing_date] = 100.000000
        next_fixing_date = max_fixing_date
        fixing_dates_no_max_date = [fixing_date for fixing_date in fixing_dates if fixing_date < max_fixing_date]
        for fixing_date in fixing_dates_no_max_date:
            yf = D.get_yf(fixing_date, next_fixing_date, 'act/360')
            fixed_factors[fixing_date] = fixed_factors[next_fixing_date] * \
                                         (1 + self.fixings[fixing_date] * yf)
            next_fixing_date = fixing_date

        return fixed_factors