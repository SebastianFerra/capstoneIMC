from Calculator import Dates as D
import math
import numpy as np
import copy

class Derivative:
    def __init__(self, operation_number, derivative_type, collateral_currency, active_leg, passive_leg):
        self.operation_number = operation_number
        self.derivative_type = derivative_type
        self.collateral_currency = collateral_currency
        self.active_leg = active_leg
        self.passive_leg = passive_leg
        self.active_leg.position = 'ACT'
        self.passive_leg.position = 'PAS'
        self.collateral_adj = self.__get_collateral_adj()
        self.__set_discount_localities()
        self.__set_discount_curves()

    def mature_to_date(self, date):
        a_l = self.active_leg
        p_l = self.passive_leg

        if self.derivative_type != 'Forward':
            a_l_c_new = [c for c in a_l.coupons if c.dates['payment_date'] >= date]
            p_l_c_new = [c for c in p_l.coupons if c.dates['payment_date'] >= date]
            if len(a_l_c_new) == 0 or len(p_l_c_new) == 0:
                return None
            a_l_new = Leg(a_l_c_new, a_l.currency)
            p_l_new = Leg(p_l_c_new, p_l.currency)
            der = Derivative(self.operation_number, self.derivative_type, self.collateral_currency, a_l_new, p_l_new)
            return der
        else:
            if a_l.coupons[0].date < date:
                return None
            else:
                return self

    def is_affected_by_curve(self, curve_name, market):
        if curve_name in ['curva_usd_cl', 'curva_usd_usa'] and self.collateral_adj:
            return True
        if curve_name in [self.active_leg.discount_curve, self.passive_leg.discount_curve]:
            return True
        if self.active_leg.coupons[0].coupon_type == 'floating':
            if market.indexes[self.active_leg.floating_index].projection_curve_name == curve_name:
                return True
        if self.passive_leg.coupons[0].coupon_type == 'floating':
            if market.indexes[self.passive_leg.floating_index].projection_curve_name == curve_name:
                return True

        return False

    def __add__(self, other):
        if (self.collateral_adj == other.collateral_adj) and (self.derivative_type == other.derivative_type):
            res_active_leg = self.active_leg + other.active_leg
            res_passive_leg = self.passive_leg + other.passive_leg
            if len(res_passive_leg) + len(res_active_leg) == 2:
                der = Derivative(1, self.derivative_type, self.collateral_currency, res_active_leg, res_passive_leg)
                return der
            else:
                return [self, other]
        else:
            return [self, other]

    def __get_collateral_adj(self):
        if self.collateral_currency == 'usd':
            return False if 'IRS' in self.derivative_type and self.active_leg.currency == 'usd' else True
        else:
            return False

    def __set_discount_localities(self):
        self.active_leg.set_discount_locality(self.derivative_type)
        self.passive_leg.set_discount_locality(self.derivative_type)

    def __set_discount_curves(self):
        self.active_leg.set_discount_curve()
        self.passive_leg.set_discount_curve()


class Leg:
    def __init__(self, coupons, currency, initialize_vectors=True):
        '''
        :param cupones: lista Cupon
        :param recibe: boolean, True para Activo, False para Pasivo
        '''
        self.position = None
        self.currency = currency
        self.coupons = coupons

        self.payment_dates = None
        self.start_dates = None
        self.end_dates = None
        self.fixing_dates = None
        self.notionals = None
        self.amortizations = None
        self.yfs = None
        self.is_floating = None
        self.floating_index = None
        self.spread_interests = None
        self.fixed_interests = None
        self.fixed_flows = None
        if initialize_vectors:
            self.__initialize_vectors()

        self.discount_locality = None
        self.discount_curve = None

    def __copy__(self):
        lg = Leg(self.coupons, self.currency, False)
        lg.discount_curve = self.discount_curve
        lg.discount_locality = self.discount_locality
        lg.payment_dates = self.payment_dates
        lg.start_dates = self.start_dates
        lg.end_dates = self.end_dates
        lg.fixing_dates = self.fixing_dates
        lg.notionals = self.notionals
        lg.amortizations = self.amortizations
        lg.yfs = self.yfs
        lg.is_floating = self.is_floating
        lg.floating_index = self.floating_index
        lg.spread_interests = self.spread_interests
        lg.fixed_interests = self.fixed_interests
        lg.fixed_flows = self.fixed_flows
        return lg

    def set_discount_locality(self, derivative_type):
        if derivative_type != 'IRS':
            self.discount_locality = 'cl'
        else:
            self.discount_locality = 'cl' if self.currency in ('clp', 'clf') else 'usa' if self.currency == 'usd' \
                                                                                        else self.currency[0:2]

    def set_discount_curve(self):
        self.discount_curve = 'curva_' + self.currency + '_' + self.discount_locality

    def strip_payed_coupons(self, date):
        length = len(self.coupons)
        i = 0
        while i < length:
            if self.coupons[i].dates['payment_date'] <= date:
                self.coupons.pop(i)
                length -= 1
                i -= 1
            i += 1
        if len(self.coupons) > 0:
            self.__initialize_vectors()


    def __add__(self, other):
        if (self.currency == other.currency) and (self.is_floating == other.is_floating):
            if self.is_floating and (self.floating_index != other.floating_index):
                return [self, other]
            else:
                self_coupons = copy.copy(self.coupons)
                other_coupons = copy.copy(other.coupons)
                self_coupon_length = len(self_coupons)
                other_coupon_length = len(other_coupons)
                new_coupons_list = []
                for i in range(0, self_coupon_length):
                    for j in range(0, other_coupon_length):
                        added_coupon = self.coupons[i] + other.coupons[j] if self.position == other.position else self.coupons[i] - other.coupons[j]
                        if type(added_coupon) is not list:
                            new_coupons_list.append(added_coupon)
                            self_coupons.pop(i)
                            other_coupons.pop(j)
                            i -= 1
                            j -= 1
                            self_coupon_length -= 1
                            other_coupon_length -= 1

                new_coupons_list.extend(self_coupons)
                new_coupons_list.extend(other_coupons)
                return Leg(new_coupons_list, self.currency)
        else:
            return [self, other]

    def __initialize_vectors(self):
        self.payment_dates = np.array([np.datetime64(coupon.dates['payment_date'])
                                       for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None
        self.start_dates = np.array([np.datetime64(coupon.dates['start_date'])
                                     for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None
        self.end_dates = np.array([np.datetime64(coupon.dates['end_date'])
                                   for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None
        self.fixing_dates = np.array([np.datetime64(coupon.dates['fixing_date'])
                                      for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None
        self.notionals = np.array([coupon.notional
                                   for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None
        self.amortizations = np.array([coupon.amortization
                                       for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None

        self.yfs = np.array([D.get_yf(coupon.dates['start_date'], coupon.dates['end_date'], coupon.day_count_convention)
                             for coupon in self.coupons]) if self.coupons[0].coupon_type != 'fx_forward' else None

        self.is_floating = False if self.coupons[0].coupon_type in ['fixed', 'fx_forward'] else True
        self.floating_index = self.coupons[0].floating_index if self.is_floating else None

        self.spread_interests = self.notionals * np.array([coupon.spread for coupon in self.coupons]) * self.yfs \
            if self.is_floating else np.zeros(len(self.coupons)) \
            if self.coupons[0].coupon_type != 'fx_forward' else None
        self.fixed_interests = self.notionals * np.array([coupon.fixed_rate for coupon in self.coupons]) * self.yfs \
            if not self.is_floating and self.coupons[0].coupon_type != 'fx_forward' \
            else np.zeros(len(self.coupons)) if self.coupons[0].coupon_type != 'fx_forward' else None
        self.fixed_flows = self.spread_interests + self.fixed_interests + self.amortizations \
            if self.coupons[0].coupon_type != 'fx_forward' else None


class FixedCoupon:
    def __init__(self, notional, amortization, dates, fixed_rate, interest_convention, day_count_convention):
        self.notional = notional
        self.dates = dates
        self.day_count_convention = day_count_convention
        self.interest_convention = interest_convention
        self.amortization = amortization
        self.coupon_type = 'fixed'

        self.fixed_rate = fixed_rate if interest_convention == 'Linear' else self.__linearize_rate(fixed_rate)

    def __linearize_rate(self, fixed_rate):
        yf = D.get_yf(self.dates['start_date'], self.dates['end_date'], self.day_count_convention)
        if self.interest_convention == 'Compounded':
            result_rate = ((1 + fixed_rate) ** yf - 1) / yf
        elif self.interest_convention == 'Exponential':
            result_rate = (math.exp(fixed_rate*yf) - 1) / yf
        else:
            raise Exception('Interest convention not recognized.')

        return result_rate

    def __from_linear_anything_to_linear_act360(self, linear_rate, day_count_convention_rate, rate_dates):
        yf = D.get_yf(rate_dates['start_date'], rate_dates['end_date'], day_count_convention_rate)
        yf_new = (rate_dates['end_date'] - rate_dates['start_date']).days / 360.0000
        new_rate = linear_rate * yf / yf_new
        return new_rate

    def __add__(self, other):
        if self.dates == other.dates:
            self_rate = self.__from_linear_anything_to_linear_act360(self.fixed_rate, self.day_count_convention, self.dates)
            other_rate = self.__from_linear_anything_to_linear_act360(other.fixed_rate, other.day_count_convention, other.dates)
            new_notional = self.notional + other.notional
            new_amortization = self.amortization + other.amortization
            new_rate = (self_rate * self.notional + other_rate * other.notional)/new_notional
            return FixedCoupon(new_notional, new_amortization, self.dates, new_rate, 'Linear', 'act/360')
        else:
            return [self, other]

    def __sub__(self, other):
        if self.dates == other.dates:
            self_rate = self.__from_linear_anything_to_linear_act360(self.fixed_rate, self.day_count_convention,
                                                                     self.dates)
            other_rate = self.__from_linear_anything_to_linear_act360(other.fixed_rate, other.day_count_convention,
                                                                      other.dates)
            new_notional = self.notional - other.notional
            new_amortization = self.amortization - other.amortization
            new_rate = (self_rate * self.notional - other_rate * other.notional) / new_notional
            return FixedCoupon(new_notional, new_amortization, self.dates, new_rate, 'Linear', 'act/360')
        else:
            return [self, other]


class FXForwardCoupon:
    def __init__(self, notional, date):
        self.notional = notional
        self.date = date
        self.coupon_type = 'fx_forward'


class FloatingCoupon:
    def __init__(self, notional, amortization, spread, dates, floating_index, interest_convention, day_count_convention):
        self.notional = notional
        self.amortization = amortization
        self.dates = dates
        self.floating_index = floating_index.lower()
        self.interest_convention = interest_convention
        self.day_count_convention = day_count_convention
        self.coupon_type = 'floating'
        self.spread = spread if interest_convention == 'Linear' else self.__linearize_rate(spread)
        self.is_overnight = True if floating_index.lower() in ['icp_clp', 'icp_clf', 'fed', 'eonia', 'ibr'] else False

    def __add__(self, other):
        if self.dates == other.dates:
            self_spread = self.spread
            other_spread = other.spread
            new_notional = self.notional + other.notional
            new_amortization = self.amortization + other.amortization
            new_spread = (self_spread * self.notional + other_spread * other.notional) / new_notional
            return FloatingCoupon(new_notional, new_amortization, new_spread, self.dates, self.floating_index, 'Linear',
                                  'act/360')
        else:
            return [self, other]

    def __sub__(self, other):
        if self.dates == other.dates:
            self_spread = self.spread
            other_spread = other.spread
            new_notional = self.notional - other.notional
            new_amortization = self.amortization - other.amortization
            new_spread = (self_spread * self.notional - other_spread * other.notional) / new_notional
            return FloatingCoupon(new_notional, new_amortization, new_spread, self.dates, self.floating_index, 'Linear',
                                  'act/360')
        else:
            return [self, other]

    def __linearize_rate(self, fixed_rate):
        yf = D.get_yf(self.dates['start_date'], self.dates['end_date'], self.day_count_convention)
        if self.interest_convention == 'Compounded':
            result_rate = ((1 + fixed_rate) ** yf - 1) / yf
        elif self.interest_convention == 'Exponential':
            result_rate = (math.exp(fixed_rate * yf) - 1) / yf
        else:
            raise Exception('Interest convention not recognized.')
        return result_rate
