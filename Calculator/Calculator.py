import numpy as np
from operator import itemgetter
import sys


def get_mtm(derivative, market, mtm_currency, print_errors=False):
    if isinstance(derivative, list):
        return sum([get_mtm(der, market, mtm_currency) for der in derivative])
    else:
        try:
            if derivative.derivative_type == 'Forward':
                return __get_forward_mtm(derivative, market, mtm_currency)
            else:
                return __get_swap_mtm(derivative, market, mtm_currency)
        except Exception as ex:
            if print_errors:
                print('No se pudo valorizar la operacion', str(derivative.operation_number) + '. Se asignó MtM = 0')
                print('\tMensaje de error:', str(sys.exc_info()[0]) + ':', ex)
            return 0


def __get_forward_mtm(fwd, market, mtm_currency):
    md = market.market_date

    act_coupon = fwd.active_leg.coupons[0]
    act_pay_days = (act_coupon.date - md).days
    active_flow = act_coupon.notional
    act_df = market.curves[fwd.active_leg.discount_curve][act_pay_days]

    pass_coupon = fwd.passive_leg.coupons[0]
    pass_pay_days = (act_coupon.date - md).days
    passive_flow = pass_coupon.notional
    pass_df = market.curves[fwd.passive_leg.discount_curve][pass_pay_days]

    if fwd.collateral_adj:
        act_df *= market.curves['col_adj_curve'][act_pay_days]
        pass_df *= market.curves['col_adj_curve'][pass_pay_days]

    act_fx = market.get_fx(fwd.active_leg.currency, mtm_currency)
    pass_fx = market.get_fx(fwd.passive_leg.currency, mtm_currency)
    return active_flow * act_df * act_fx - passive_flow * pass_df * pass_fx


def __get_swap_mtm(swap, market, mtm_currency):
    md = market.market_date
    md_np = np.datetime64(md)

    active_leg = swap.active_leg
    active_payment_days = (active_leg.payment_dates - md_np).astype('timedelta64[D]').astype(int)
    active_flows = active_leg.fixed_flows.copy()
    active_dfs = np.array(itemgetter(*active_payment_days)(market.curves[active_leg.discount_curve]))

    passive_leg = swap.passive_leg
    passive_payment_days = (passive_leg.payment_dates - md_np).astype('timedelta64[D]').astype(int)
    passive_flows = passive_leg.fixed_flows.copy()
    passive_dfs = np.array(itemgetter(*passive_payment_days)(market.curves[passive_leg.discount_curve]))


    if swap.collateral_adj:
        col_adj_curve = market.curves['col_adj_curve']
        active_dfs = active_dfs * np.array(itemgetter(*active_payment_days)(col_adj_curve))
        passive_dfs = passive_dfs * np.array(itemgetter(*passive_payment_days)(col_adj_curve))

    if active_leg.is_floating:
        active_flows = active_flows + __get_floating_interest(active_leg, md, market)
    if passive_leg.is_floating:
        passive_flows = passive_flows + __get_floating_interest(passive_leg, md, market)

    active_fx = market.get_fx(swap.active_leg.currency, mtm_currency)
    passive_fx = market.get_fx(swap.passive_leg.currency, mtm_currency)

    active_mtm = np.sum(active_flows * active_dfs)
    passive_mtm = np.sum(passive_flows * passive_dfs)

    return active_fx * active_mtm - passive_fx * passive_mtm


def __get_floating_interest(leg, market_date, market):
    index = market.indexes[leg.floating_index]
    projection_df_curve = market.curves[index.projection_curve_name]

    coupons_fl = [coupon for coupon in leg.coupons if coupon.dates['fixing_date' if not coupon.is_overnight else 'start_date'] > market_date]
    fixed_coupons_amount = len(leg.coupons) - len(coupons_fl)

    res_flows = np.array([])
    if fixed_coupons_amount > 0:
        started_coupons = [coupon for coupon in leg.coupons if coupon.dates['fixing_date' if not coupon.is_overnight else 'start_date'] <= market_date]
        if started_coupons[0].is_overnight:
            factors = index.fixed_factors
            df_fins_sc = np.array(
                [projection_df_curve[(coupon.dates['end_date'] - market_date).days] for coupon in started_coupons])
            semifloat_factors = (np.array([factors[started_coupon.dates['start_date']] / factors[market_date]
                                      for started_coupon in started_coupons])) / df_fins_sc - 1
            started_flows = semifloat_factors * leg.notionals[:fixed_coupons_amount]
        else:
            fixings = index.fixings
            fixed_rates = np.array([fixings[started_coupon.dates['fixing_date']] for started_coupon in started_coupons])
            started_flows = fixed_rates * leg.yfs[:fixed_coupons_amount] * leg.notionals[:fixed_coupons_amount]
        res_flows = started_flows

    if len(coupons_fl) > 0:
        start_date_tenors = [(coupon.dates['start_date'] - market_date).days for coupon in coupons_fl]
        end_date_tenors = [(coupon.dates['end_date'] - market_date).days for coupon in coupons_fl]
        df_inis = np.array(itemgetter(*start_date_tenors)(projection_df_curve))
        df_fins = np.array(itemgetter(*end_date_tenors)(projection_df_curve))
        future_flows = leg.notionals[fixed_coupons_amount:] * (df_inis / df_fins - 1)
        res_flows = np.concatenate([res_flows, future_flows])

    return res_flows
