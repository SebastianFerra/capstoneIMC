import numpy as np
import QuantLib as ql
from tqdm import tqdm 

#for testing
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import xlwings as xw
import seaborn as sns



class Swap:
    def __init__(self,years,side):
        self.y = years
        self.side = side
        self.cupon = 0

    def set_eval_date(self, t):
        self.t=t
        #Condiciones originales
        self.yf = np.arange(0.5,self.y+0.5,0.5)
        if t in self.yf:
            self.cupon = 1
        else:
            self.cupon = 0
        self.acc_yf = np.full(self.y*2,0.5)
        self.yf = self.yf - t
        time = (self.yf>0)
        self.yf = self.yf[time]
        paid = np.size(time)-np.count_nonzero(time)
        self.acc_yf = self.acc_yf[time]
        return 0
    @property
    def fixed_rate(self):
        return self._fixed_rate
    @fixed_rate.setter
    def fixed_rate(self, arg):
        self._fixed_rate = arg


class IRS(Swap):
    def __init__(self, years,model,n=1,side=1,fixed_rate=None):
        super().__init__(years,side)
        self._fixed_rate = fixed_rate
        self.model = model
        self.n = n
    def get_dfs(self, r0):
        self.df_n = self.model.zero_cupon(self.t,self.t+self.yf-0.5, r0)
        self.df_n_1 = self.model.zero_cupon(self.t,self.t+self.yf, r0)
        if self.cupon == 1 or self.t==0:
            self.n_r = (self.df_n/self.df_n_1-1)/0.5
            #self.n_r = ((1/self.df_n)-1)*1/self.yf 

        elif self.cupon==0:
            if self.df_n.ndim>1:
                r_i = self.n_r[:,0]
                r_i = r_i.reshape(r_i.size,1)
                #r_a = ((1/self.df_n[:,1:])-1)*1/self.yf[1:]
                r_a = ((self.df_n[:,1:]/self.df_n_1[:,1:])-1)/0.5
                self.n_r = np.append(r_i, r_a,axis=1)
            else:
                r_i = np.array([self.n_r[0]])
                #r_a = ((1/self.df_n[1:])-1)*1/self.yf[1:]
                r_a = ((self.df_n[1:]/self.df_n_1[1:])-1)/0.5
                self.n_r = np.concatenate([r_i, r_a])
        return 0
    def get_rate(self, r0, set_rate=False):
        self.get_dfs(r0)
        r = (self.n_r*self.df_n*self.acc_yf).sum()/(self.df_n*self.acc_yf).sum()
        if set_rate==True:
            self._fixed_rate = r
        return r
    def get_mtm(self,r0):
        if self.t>=self.y:
            return 0
        self.get_dfs(r0)
        if self.df_n.ndim==1:
            self.fix_l = self.n*(self._fixed_rate*np.sum(self.df_n*self.acc_yf)+self.df_n[-1])
            self.float_l = self.n*(np.sum(self.n_r*self.df_n*self.acc_yf)+self.df_n[-1])
        else:
            self.fix_l = self.n*(self._fixed_rate*np.sum(self.df_n*self.acc_yf,axis=1)+self.df_n[:,-1])
            self.float_l = self.n*(np.sum(self.n_r*self.df_n*self.acc_yf,axis=1)+self.df_n[:,-1])
        if self.side==1:
            return self.float_l-self.fix_l
        else:
            return self.fix_l-self.float_l

    def get_dur(self):
        vp = self.n*(self._fixed_rate*np.sum(self.df_n*self.acc_yf)+self.df_n[-1])
        vp01 = self.n*((self._fixed_rate+(0.01/100))*np.sum(self.df_n*self.acc_yf)+self.df_n[-1])
        return (vp01-vp)*(1/(0.01/100))/self.n


class CrossUFICP(Swap):
    def __init__(self,years,models,UF_spot,n=1,side=1,fixed_rate=None):
        super().__init__(years,side)
        self.n_model = models[0]
        self.r_model = models[1]
        self._fixed_rate = fixed_rate
        self.UF_spot = UF_spot
        self.n = n
    def get_dfs(self,r0,uf0):
        n = r0[0]
        r = r0[1]
        self.df_n = self.n_model.zero_cupon(self.t,self.t+self.yf-0.5, n)
        self.df_n_1 = self.n_model.zero_cupon(self.t,self.t+self.yf, n)
        self.df_r = self.r_model.zero_cupon(self.t,self.t+self.yf, r)


        if self.cupon == 1 or self.t==0:
            #self.proy_n = (1/self.df_n-1)/self.yf
            self.proy_n = ((self.df_n/self.df_n_1)-1)/0.5
            self.proy_i = ((self.df_r/self.df_n).T*uf0).T
        elif self.cupon == 0:
            a = self.proy_n[:,0]
            b = self.proy_i[:,0]
            a = a.reshape(a.shape[0],1)
            b = b.reshape(b.shape[0],1)
            #self.proy_n = (1/self.df_n[:,1:]-1)/self.yf[1:]
            self.proy_n = ((self.df_n[:,1:]/self.df_n_1[:,1:])-1)/0.5
            self.proy_i = ((self.df_r[:,1:]/self.df_n[:,1:]).T*uf0).T

            self.proy_n = np.append(a, self.proy_n,axis=1)
            self.proy_i = np.append(b, self.proy_i,axis=1)
            #uf_c = self.proy_i*self.yf+1
            #self.proy_i = (uf_c.T*uf0).T
        return 0
    def get_rate(self,r0,uf0,n_rate,set_rate=False):
        self.n_rate = n_rate
        self.get_dfs(r0,uf0)
        a = (self.UF_spot-self.proy_i[-1])*self.df_n[-1]+n_rate*np.sum(self.df_n*self.acc_yf)*self.UF_spot
        b = np.sum(self.proy_i*self.df_n*self.acc_yf)
        if set_rate==True:
            self._fixed_rate=a/b
        return a/b
    def get_mtm(self, r0, uf0):
        if self.t>=self.y:
            return 0
        self.get_dfs(r0, uf0)
        if self.df_n.ndim==1:
            self.n_l = self.n*self.UF_spot*(np.sum(self.proy_n*self.df_n*self.acc_yf)+self.df_n[-1])
            self.r_l = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        else:
            self.n_l = self.n*self.UF_spot*(np.sum(self.proy_n*self.df_n*self.acc_yf,axis=1)+self.df_n[:,-1])
            self.r_l = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf,axis=1)+self.df_n[:,-1]*self.proy_i[:,-1])
        if self.side==1:
            return self.n_l-self.r_l
        else:
            return self.r_l-self.n_l
    def get_dur(self):
        vp = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        vp01 = self.n*((self._fixed_rate+0.01/100)*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        return (vp01-vp)*(1/(0.01/100))/self.n


class Basis(Swap):
    def __init__(self,years,models,spot,n=1,side=1,spread=0):
        super().__init__(years,side)
        self.n_model = models[0]
        self.l_model = models[1]
        self.usd_loc_model = models[2]
        self.spread = spread
        self.spot = spot
        self.n = n
    def get_dfs(self,r0, spot):
        n = r0[0]
        l = r0[1]
        usd_loc = r0[2]
        self.df_n = self.n_model.zero_cupon(self.t,self.t+self.yf, n)
        self.df_l = self.l_model.zero_cupon(self.t,self.t+self.yf, l)

        df_n_1 = self.n_model.zero_cupon(self.t,self.t+self.yf+0.5, n)
        df_l_1 = self.l_model.zero_cupon(self.t,self.t+self.yf+0.5, l)
        self.df_usd_loc = self.usd_loc_model.zero_cupon(self.t,self.t+self.yf, usd_loc)

        if self.cupon == 1 or self.t==0:
            #self.proy_n = (1/self.df_n-1)/self.yf
            #self.proy_l = (1/self.df_l-1)/self.yf
            self.proy_n = ((self.df_n/df_n_1)-1)/0.5
            self.proy_l = ((self.df_l/df_l_1)-1)/0.5
        elif self.cupon == 0:
            a = self.proy_n[:,0]
            b = self.proy_l[:,0]
            a = a.reshape(a.shape[0],1)
            b = b.reshape(b.shape[0],1)
            #self.proy_n = (1/self.df_n[:,1:]-1)/self.yf[1:]
            #self.proy_l = (1/self.df_l[:,1:]-1)/self.yf[1:]
            self.proy_n = ((self.df_n[:,1:]/df_n_1[:,1:])-1)/0.5
            self.proy_l = ((self.df_l[:,1:]/df_l_1[:,1:])-1)/0.5

            self.proy_n = np.append(a, self.proy_n,axis=1)
            self.proy_l = np.append(a, self.proy_l,axis=1)
        return 0
    def get_spread(self,r0,spot,set_rate=False):
        self.get_dfs(r0, spot)
        l_n = np.sum(self.proy_n*self.acc_yf*self.df_n)+self.df_n[-1]
        l_l = np.sum(self.proy_l*self.acc_yf*self.df_usd_loc)+self.df_usd_loc[-1]
        b = np.sum(self.acc_yf*self.df_usd_loc)
        if set_rate==True:
            self.spread=(l_n-l_l)/b
        return (l_n-l_l)/b
    def get_mtm(self, r0, spot):
        if self.t>=self.y:
            return 0
        self.get_dfs(r0, spot)
        if self.df_n.ndim==1:
            self.n_l = self.n*self.spot*(np.sum(self.proy_n*self.df_n*self.acc_yf)+self.df_n[-1])
            self.l_l = self.n*spot*(np.sum((self.proy_l+self.spread)*self.df_usd_loc*self.acc_yf)+self.df_usd_loc[-1])
        else:
            self.n_l = self.n*self.spot*(np.sum(self.proy_n*self.df_n*self.acc_yf,axis=1)+self.df_n[:,-1])
            self.l_l = self.n*spot*(np.sum((self.proy_l+self.spread)*self.df_usd_loc*self.acc_yf,axis=1)+self.df_usd_loc[:,-1])
        if self.side==1:
            return self.n_l-self.l_l
        else:
            return self.l_l-self.n_l
    def get_dur(self):
        vp = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        vp01 = self.n*((self._fixed_rate+0.01/100)*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        return (vp01-vp)*(1/(0.01/100))/self.n


class Forward:
    def __init__(self,years,side,models,spot,n=1):
        self.y = years
        self.side = side
        self.fwd_price = fwd_price
        self.n_model = models[0]
        self.usd_loc_model = models[1]
        self.n = n
    def set_eval_date(self, t):
        self.t=t
        #Condiciones originales
        self.yf = self.y - t
        return 0
    @property
    def fixed_rate(self):
        return self._fixed_rate
    @fixed_rate.setter
    def fixed_rate(self, arg):
        self.fwd_price = arg
    def get_dfs(self,r0):
        n = r0[0]
        usd_loc = r0[1]
        self.df_n = self.n_model.zero_cupon(self.t,self.t+self.yf, n)
        self.df_usd_loc =  self.usd_loc_model.zero_cupon(self.t,self.t+self.yf, usd_loc)
    def get_price(self,r0,spot):
        pass
    def get_mtm(self,r0, spot):
        pass


class CrossUFUSD(Swap):
    def __init__(self,years,models,spots,n=1,side=1,fixed_rate=None):
        super().__init__(years,side)
        self.n_model = models[0]
        self.r_model = models[1]
        self.l_model = models[2]
        self.usd_loc_model = models[3]

        self._fixed_rate = fixed_rate

        self.UF_spot = spots[0]
        self.usd_spot = spots[1]

        self.n = n
    def get_dfs(self,r0,uf0):
        n = r0[0]
        r = r0[1]
        l = r0[2]
        usd_loc = r0[3]

        #Dolar
        self.df_usd_loc = self.usd_loc_model.zero_cupon(self.t,self.t+self.yf, usd_loc)
        self.df_l = self.l_model.zero_cupon(self.t,self.t+self.yf, l)
        self.df_l_1 = self.l_model.zero_cupon(self.t,self.t+self.yf+0.5, l)
        #Peso
        self.df_n = self.n_model.zero_cupon(self.t,self.t+self.yf, n)
        self.df_r = self.r_model.zero_cupon(self.t,self.t+self.yf, r)

        #UF Proyectada
        self.proy_i = ((self.df_r/self.df_n).T*uf0).T
        if self.cupon == 1 or self.t==0:
            self.proy_l = ((self.df_l/self.df_l_1)-1)/0.5
        elif self.cupon == 0:
            a = self.proy_l[:,0]
            a = a.reshape(a.shape[0],1)
            self.proy_l = ((self.df_l[:,1:]/self.df_l_1[:,1:])-1)/0.5
            self.proy_l = np.append(a, self.proy_l,axis=1)
        return 0
    def get_rate(self,r0,spots,set_rate=False):
        uf0 = spots[0]
        usd = spots[1]
        self.get_dfs(r0,uf0)

        #uf_leg = ((self.df_n*self.proy_i*self.acc_yf) + self.df_n[-1]*self.proy_i[-1])/self.UF_spot
        libor_leg = np.sum((self.df_usd_loc*self.proy_l*self.acc_yf)) + self.df_usd_loc[-1]
        a = libor_leg*self.UF_spot - self.df_n[-1]*self.proy_i[-1]
        b = np.sum(self.df_n*self.proy_i*self.acc_yf)
        if set_rate==True:
            self._fixed_rate=a/b
        return a/b
    def get_mtm(self, r0, spots):
        uf0 = spots[0]
        usd =  spots[1]
        if self.t>=self.y:
            return 0
        self.get_dfs(r0, uf0)
        if self.df_n.ndim==1:
            self.l_l = self.n*usd*self.UF_spot/self.usd_spot*(np.sum(self.proy_l*self.df_usd_loc*self.acc_yf)+self.df_usd_loc[-1])
            self.r_l = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        else:
            self.l_l = self.n*usd*self.UF_spot/self.usd_spot*(np.sum(self.proy_l*self.df_usd_loc*self.acc_yf,axis=1)+self.df_usd_loc[:,-1])
            self.r_l = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf,axis=1)+self.df_n[:,-1]*self.proy_i[:,-1])
        if self.side==1:
            return self.r_l-self.l_l
        else:
            return self.l_l-self.r_l
    def get_dur(self):
        vp = self.n*(self._fixed_rate*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        vp01 = self.n*((self._fixed_rate+0.01/100)*np.sum(self.proy_i*self.df_n*self.acc_yf)+self.df_n[-1]*self.proy_i[-1])
        return (vp01-vp)*(1/(0.01/100))/self.n


class CurvaDescuento(ql.CubicZeroCurve):
    def __init__(self, dates, rates, day_count):
        super().__init__(dates,rates,day_count, ql.TARGET(),ql.Cubic(),ql.Simple)
    def vec_discount(self, t):
        if isinstance(t,np.ndarray):
            return np.vectorize(self.discount)(t)
        else:
            return self.discount(t)
    def vec_forwardRate(self,t,T):
        pt = self.vec_discount(t)
        pT = self.vec_discount(T)
        return  np.log(pt/pT)/(T-t)


#________Modelos_____________


class HullWhite_1F:
    def __init__(self, dt, alpha, sigma, spot_curve, r0=None):
        self.alpha = alpha
        self.sigma = sigma
        self.curve = spot_curve
        self.dt = dt
        self.r0 = r0
    def zero_cupon(self,t,T,r0):
        #brigo-mercurio
        pT = self.curve.vec_discount(T)
        pt = self.curve.vec_discount(t)
        ft = self.curve.vec_forwardRate(t,t+self.dt)
        B = lambda t_, T_: (1/self.alpha)*(1-np.exp(-self.alpha*(T_-t_)))
        b = B(t,T)
        tmp = b*self.sigma
        #a = (pT/pt)*np.exp(b*ft-(((self.sigma**2)/(4*self.alpha))*(1-np.exp(-2*self.alpha*t))*b**2))
        value = B(t,T)*ft - 0.25*(tmp**2)*B(0,2*t)
        a = (pT/pt)*np.exp(value)
        if isinstance(r0,np.ndarray):
            tmp = np.exp(np.outer(-b,r0))
            return np.multiply(a,tmp.T)
        else:
            return a*np.exp(-b*r0)
    def theta(self,t):
        f0 = self.curve.vec_forwardRate(t-self.dt,t)
        f1 = self.curve.vec_forwardRate(t,t+self.dt)
        df = (f1-f0)/(2*self.dt)
        B = lambda t_, T_: (1/self.alpha)*(1-np.exp(-self.alpha*(T_-t_)))
        return df+f0*self.alpha+0.5*(self.sigma**2)*B(0,2*t)


class ConstantSpreadHW(HullWhite_1F):
    def __init__(self, base_model, spread_vector):
        self.base_model = base_model
        self.spread_vector = spread_vector
        self.sigma = base_model.sigma
        self.alpha = base_model.alpha
        self.dt = base_model.dt
        self.r0 = base_model.r0
        self.curve = base_model.curve


def correlated_BM(cov,steps,n_sim=100,T=3000):
    n_sde = cov.shape[0]
    L = np.linalg.cholesky(cov)
    W = np.random.normal(0,1,size=(T,int(n_sim/2),n_sde))
    W = np.append(W,W*-1,axis=1)
    for i in range(W.shape[1]):
        W[:,i,:] = (L @ W[:,i,:].T).T
    return W


def testing():
    from scipy import random, linalg
    import time

    matrixSize = 6
    A = random.rand(matrixSize,matrixSize)
    cov = np.dot(A,A.transpose())

    dt=1/250

    start = time.time()
    correlated_BM_2(cov,dt,n_sim=10000,T=3000)
    end = time.time()
    print("Elapsed (without numba) = %s" % (end - start))

    start = time.time()
    L = np.linalg.cholesky(cov)
    n_sde = cov.shape[0]
    correlated_BM(L,dt,n_sde,n_sim=10000,T=3000)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    start = time.time()
    L = np.linalg.cholesky(cov)
    n_sde = cov.shape[0]
    correlated_BM(L,dt,n_sde, n_sim=10000,T=3000)
    end = time.time()
    print("Elapsed (without compilation) = %s" % (end - start))
if __name__ == '__main__':
    testing2()