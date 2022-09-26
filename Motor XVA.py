import numpy as np
import QuantLib as ql
from tqdm import tqdm 

#for testing
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import xlwings as xw
import seaborn as sns

from Calculator import Calculator as Calc
from datetime import datetime as dt
import Factory as Fct
import pickle
import pandas as pd

#Functions
from functions_xva import *
pd.options.display.float_format = '{:,.4f}'.format
np.set_printoptions(precision=4)

def get_inputs(market_date):
    inputs_csv_delimiter = ';'
    inputs_decimal_separator = ','

    print('Cargando data de mercado...')
    data = pd.read_csv('data.csv', delimiter=inputs_csv_delimiter, decimal=inputs_decimal_separator)
    vol = pd.read_csv('vol.csv', delimiter=inputs_csv_delimiter, decimal=inputs_decimal_separator)
    data_historica = pd.read_csv('history.csv', delimiter=inputs_csv_delimiter, decimal=inputs_decimal_separator)
    
    return data, vol, data_historica
    
def get_index(name, matrix):
    for i, n in enumerate(matrix.columns):
        if n==name:
            return i
    return None


market_date = dt(2022, 7, 15)    
Today = market_date 
TODAY = ql.Settings.instance().evaluationDate
ANOS_SIMULADOS = 20
DIAS_POR_ANO = 250
N_SIM = 100

DT = 1/DIAS_POR_ANO
T = ANOS_SIMULADOS*DIAS_POR_ANO
STEPS = np.arange(1,T+1)*DT


data, vol, data_historica = get_inputs(market_date)

SIGMA_NOMINAL = vol['icp_vol'][0]
SIGMA_REAL = vol['real_vol'][0]
SIGMA_INFLACION = vol['infla_vol'][0]
SIGMA_TC = vol['tc_vol'][0]
SIGMA_LIBOR = vol['libor_vol'][0]
SIGMA_USD_LOCAL = vol['usd_loc_vol'][0]
SIGMA_OIS = vol['ois_vol'][0]

UF_HOY = vol['uf_spot'][0]
USD_HOY = vol['tc_spot'][0]

print('UF Hoy: ', UF_HOY)
print('SPOT Hoy: ', USD_HOY)

#CALCULO DE RETORNO DEL TIPO DE CAMBIO
data_historica['tc_log_r']= np.log(data_historica.tc).diff()*100
data_historica.drop(columns=['tc'],inplace=True)
data_historica.dropna(inplace=True)

CORR_MATRIX = data_historica.corr()
COV_MATRIX = data_historica.cov()

print(COV_MATRIX)

W = correlated_BM(COV_MATRIX, STEPS ,n_sim=N_SIM,T=T)


#Curva CLP

year_fractions = data['YF(360)'].values
discount_nominal = data['DF CLP_CL'].values
dates = [TODAY+int(x) for x in data.Dias.values]
tasas_nominal = (1/discount_nominal-1)/year_fractions

dates.insert(0,TODAY)
tasas_nominal = np.insert(tasas_nominal, 0,0)
curva_nominal = CurvaDescuento(dates, tasas_nominal.tolist(), ql.Actual360())


#Curva Zero Real
uf_forward = data['UF Seguro USD'].values
pi = uf_forward/UF_HOY
discount_real = discount_nominal*pi

tasas_real = (1/discount_real-1)*1/year_fractions
tasas_real = np.insert(tasas_real, 0, 0)
tasas_real[1] = tasas_real[2]
curva_real = CurvaDescuento(dates, tasas_real.tolist(), ql.Actual360())


#Curva Libor

discount_libor = data['DF Libor'].values
tasas_libor = (1/discount_libor[:-1]-1)*1/year_fractions[1:]
tasas_libor = np.insert(tasas_libor, 0,0)
dates_libor = [dates[0]] + dates[2:]
curva_libor = CurvaDescuento(dates_libor, tasas_libor.tolist(), ql.Actual360())

#Curva USD Local

discount_usd_clp = data['DF USD_CLP'].values
tasas_usd_local = (1/discount_usd_clp-1)/year_fractions
tasas_usd_local = np.insert(tasas_usd_local, 0,0)
curva_usd_local = CurvaDescuento(dates, tasas_usd_local.tolist(), ql.Actual360())

#Curva USD Local
discount_ois = data['DF SOFR'].values
tasas_ois = (1/discount_ois-1)/year_fractions
tasas_ois = np.insert(tasas_ois, 0,0)
curva_ois = CurvaDescuento(dates, tasas_ois.tolist(), ql.Actual360())


#Curva LOIS
LOIS = np.zeros(shape=T+1)
LOIS[1:] = curva_libor.vec_forwardRate(STEPS-DT, STEPS)- curva_ois.vec_forwardRate(STEPS-DT, STEPS)
LOIS[0] = LOIS[1]

#Output QL
tasas_nominal_ql = [curva_nominal.zeroRate(y, ql.Simple, ql.Annual).rate() for y in year_fractions]
tasas_real_ql = [curva_real.zeroRate(y, ql.Simple, ql.Annual).rate() for y in year_fractions]
tasas_libor_ql = [curva_libor.zeroRate(y, ql.Simple, ql.Annual).rate() for y in year_fractions]
tasas_usd_local_ql = [curva_usd_local.zeroRate(y, ql.Simple, ql.Annual).rate() for y in year_fractions]
tasas_ois_ql = [curva_ois.zeroRate(y, ql.Simple, ql.Annual).rate() for y in year_fractions]
#Curva Forward
t = np.arange(0.5,10.5,0.1)
forward_nominal = curva_nominal.vec_forwardRate(t-0.5,t)
forward_real = curva_real.vec_forwardRate(t-0.5,t)
forward_ois = curva_ois.vec_forwardRate(t-0.5,t)


#Alpha
ALPHA_NOMINAL = 0.1
ALPHA_REAL = 0.1
ALPHA_LIBOR = 0.1
ALPHA_USD_LOCAL = 0.1
ALPHA_OIS = 0.1

#modelos
r0_nominal = curva_nominal.zeroRate(DT, ql.Simple).rate()
HW_nominal = HullWhite_1F(DT, ALPHA_NOMINAL, SIGMA_NOMINAL, curva_nominal, r0_nominal)

r0_real = curva_real.zeroRate(DT, ql.Simple).rate()
HW_real = HullWhite_1F(DT, ALPHA_REAL,SIGMA_REAL, curva_real, r0_real)

#pendiente: cambiar sigma de ois
r0_ois = curva_ois.zeroRate(DT, ql.Simple).rate()
HW_ois = HullWhite_1F(DT, ALPHA_OIS, SIGMA_LIBOR, curva_ois, r0_ois)

r0_libor = curva_libor.zeroRate(DT, ql.Simple).rate()

theta_nominal = HW_nominal.theta(STEPS)
theta_real = HW_real.theta(STEPS)
theta_ois = HW_ois.theta(STEPS)

#Fit Tasa Cero
t = np.arange(0.5,10.5,0.5)

HW_zeros_nominal = (1/HW_nominal.zero_cupon(0,t,r0_nominal)-1)/t
tasas_nominal = [curva_nominal.zeroRate(y, ql.Simple, ql.Annual).rate() for y in t]

HW_zeros_real = (1/HW_real.zero_cupon(0,t,r0_real)-1)/t
tasas_real = [curva_real.zeroRate(y, ql.Simple, ql.Annual).rate() for y in t]

HW_zeros_ois = (1/HW_ois.zero_cupon(0,t,r0_ois)-1)/t
tasas_ois = [curva_ois.zeroRate(y, ql.Simple, ql.Annual).rate() for y in t]

#Monte Carlo
#Inicializacion matrices
n = np.zeros(shape=(T+1, W.shape[1]))
dn = np.zeros(shape=(T, W.shape[1]))
#real
r = np.zeros(shape=(T+1, W.shape[1]))
dr = np.zeros(shape=(T, W.shape[1]))
#infla
i = np.zeros(shape=(T+1, W.shape[1]))
di = np.zeros(shape=(T, W.shape[1]))
#ois (como L)
ois = np.zeros(shape=(T+1, W.shape[1]))
dois = np.zeros(shape=(T, W.shape[1]))
#tc
tc = np.zeros(shape=(T+1, W.shape[1]))
dtc = np.zeros(shape=(T, W.shape[1]))
#Condiciones iniciales
n[0] = r0_nominal
r[0] = r0_real
ois[0] = r0_ois
tc[0] = USD_HOY
i[0] = UF_HOY

#Variables aleatorias
dw_nominal = W[:,:,get_index('icp',COV_MATRIX)]
dw_real = W[:,:,get_index('real',COV_MATRIX)]
dw_inflacion = W[:,:,get_index('inflation',COV_MATRIX)]
dw_tc = W[:,:,get_index('tc_log_r',COV_MATRIX)]
dw_ois = W[:,:,get_index('ois',COV_MATRIX)]

#Precomputo
sqrt_dt = np.sqrt(DT)

dw_nominal =  dw_nominal*SIGMA_NOMINAL*sqrt_dt
dw_real =  dw_real*SIGMA_REAL*sqrt_dt
dw_inflacion =  dw_inflacion*SIGMA_INFLACION*sqrt_dt
dw_tc = dw_tc*SIGMA_TC*sqrt_dt
dw_ois =  dw_ois*SIGMA_LIBOR*sqrt_dt
#arreglar OIS y Libor
for day in range(T):
    #ois
    dois[day,:] = (theta_ois[day]-ALPHA_OIS*ois[day,:]-COV_MATRIX.at['ois','tc_log_r']/100)*DT+dw_ois[day,:]
    ois[day+1,:] = ois[day,:]+dois[day,:]
    
    #icp
    dn[day,:] = (theta_nominal[day]-ALPHA_NOMINAL*n[day,:])*DT+dw_nominal[day,:]
    n[day+1,:] = n[day,:]+dn[day,:]    
    
    #tc
    dtc[day,:] = tc[day,:]*((n[day,:]-ois[day,:])*DT+dw_tc[day,:])
    tc[day+1,:] = tc[day,:]+dtc[day,:]
    
    #real
    dr[day,:] = (theta_real[day]-ALPHA_REAL*r[day,:]-COV_MATRIX.at['real','inflation']/100)*DT+dw_real[day,:]
    r[day+1,:] = r[day,:]+dr[day,:]    
    
    #infla
    di[day,:] = i[day,:]*(n[day,:]-r[day,:])*DT+i[day,:]*dw_inflacion[day,:]
    i[day+1,:] = i[day,:]+di[day,:]

#Libor
l = ois+LOIS.reshape(-1,1)
HW_libor = HW_ois
HW_libor.curve = curva_libor

print(n)

#PROMEDIOS
avg_nominal = n.mean(axis=1)
avg_real = r.mean(axis=1)
avg_ois = ois.mean(axis=1)
avg_tc = tc.mean(axis=1)
avg_i = i.mean(axis=1)

t = np.arange(1,avg_nominal.shape[0],1)

tasas_nominal = [curva_nominal.zeroRate(y/250, ql.Simple, ql.Annual).rate() for y in t]
tasas_real = [curva_real.zeroRate(y/250, ql.Simple, ql.Annual).rate() for y in t]
tasas_ois = [curva_ois.zeroRate(y/250, ql.Simple, ql.Annual).rate() for y in t]

#MONTECARLO Covergence
avg_descuento_nominal = np.mean(np.cumprod(np.exp(-n*DT),axis=0),axis=1)
avg_nominal = ((1/avg_descuento_nominal[:-1]-1)/STEPS)

avg_descuento_real = np.mean(np.cumprod(np.exp(-r*DT),axis=0),axis=1)
avg_real = ((1/avg_descuento_real[:-1]-1)/STEPS)

avg_descuento_ois = np.mean(np.cumprod(np.exp(-ois*DT),axis=0),axis=1)
avg_ois = ((1/avg_descuento_ois[:-1]-1)/STEPS)
###################################################################################
###################################################################################
###################################################################################
###################################################################################

# Este MTM se cambia al archivo MTM.py ()
#MTM Swap

clp_models = [HW_nominal, HW_real]
clp_r0 = [r0_nominal,r0_real]

years = np.arange(2,30,2)
irs_tasas = []
cross_tasas = []
for y in years:
    side = 0
    swap = IRS(y,clp_models[0], n=1, side=side)
    swap.set_eval_date(0)
    irs_tasas.append(swap.get_rate(r0_nominal, set_rate=True)*100)
    cross = CrossUFICP(y,clp_models,UF_HOY,n=1/UF_HOY, side=side)
    cross.set_eval_date(0)
    cross_tasas.append(cross.get_rate(clp_r0,UF_HOY,swap.fixed_rate,set_rate=True)*100)
    
clp_r0 = [r0_nominal,r0_real]
basis_r0 = [r0_nominal, r0_libor, r0_libor]
cross_libor_r0 = [r0_nominal,r0_real,r0_libor,r0_libor]

########
years=[2,5,10,20]
results = {}
results_EFV = {}
clp_models = [HW_nominal,HW_real]
basis_models = [HW_nominal,HW_libor,HW_libor]
cross_libor_models = [HW_nominal, HW_real, HW_libor, HW_libor]
side = 1
for y in years:
    swap = IRS(y,clp_models[0], n=1, side=side)
    cross = CrossUFICP(y,clp_models,UF_HOY,n=1/UF_HOY, side=side)
    basis = Basis(y,models=basis_models,spot=USD_HOY,n=1/USD_HOY,side=side)
    cross_libor = CrossUFUSD(y,models=cross_libor_models,spots=[UF_HOY,USD_HOY],n=1/UF_HOY,side=side)

    swap.set_eval_date(0)
    cross.set_eval_date(0)
    basis.set_eval_date(0)
    cross_libor.set_eval_date(0)
    
    swap.get_rate(r0_nominal, set_rate=True)
    cross.get_rate(clp_r0,UF_HOY,swap.fixed_rate,set_rate=True)
    basis.get_spread(basis_r0,USD_HOY,set_rate=True)
    cross_libor.get_rate(cross_libor_r0,spots=[UF_HOY,USD_HOY],set_rate=True)
    
    
    irs_mtm = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    cross_mtm = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    basis_mtm = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    cross_libor_mtm = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    
    irs_EFV = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    cross_EFV = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    basis_EFV = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    cross_libor_EFV = np.zeros(shape=(DIAS_POR_ANO*y+1,W.shape[1]))
    ######## hASTA AQUi devuelve lo mismo que el MTM
    for x in range(DIAS_POR_ANO*y):
        t = x*DT
        swap.set_eval_date(t)
        cross.set_eval_date(t)
        basis.set_eval_date(t)
        cross_libor.set_eval_date(t)
        
        if x==0:
            discount=1
        else:    
            discount = np.exp(n[0:x,:].mean(axis=0)*-t)

        eval_r_clp = [n[x,:],r[x,:]]
        uf = i[x,:]
        irs_mtm[x,:] = swap.get_mtm(eval_r_clp[0])
        cross_mtm[x,:] = cross.get_mtm(eval_r_clp,uf)
        
        irs_EFV[x,:] = discount*irs_mtm[x,:]
        cross_EFV[x,:] = discount*cross_mtm[x,:]

        eval_r_usd = [n[x,:],l[x,:],l[x,:]]
        spot_usd = tc[x,:]
        basis_mtm[x,:] = basis.get_mtm(eval_r_usd,spot_usd)
                                                
        basis_EFV[x,:] = discount*basis_mtm[x,:]
        
        eval_r_cross_libor = [n[x,:],r[x,:],l[x,:],l[x,:]]
        spots = [i[x,:],tc[x,:]]
        cross_libor_mtm[x,:] = cross_libor.get_mtm(eval_r_cross_libor,spots)
        cross_libor_EFV[x,:] = discount*cross_libor_mtm[x,:]
        
    results[y] = {'IRS_MTM':irs_mtm,'CROSS_MTM':cross_mtm,'BASIS_MTM':basis_mtm,'CROSS_LIBOR_MTM':cross_libor_mtm}
    results_EFV[y] = {'IRS_EFV':irs_EFV,'CROSS_EFV':cross_EFV,'BASIS_EFV':basis_EFV,'CROSS_LIBOR_EFV':cross_libor_EFV}

#E MTM
#divide
mtm_pos = {}
mtm_neg = {}
for k,mtms in results.items():
    pos = {}
    neg = {}
    for w,mtm in mtms.items():
        tmp = mtm.copy()
        tmp[tmp<0] = 0
        pos[w] = tmp.mean(axis=1)

        tmp = mtm.copy()
        tmp[tmp>0] = 0
        neg[w] = tmp.mean(axis=1)
    mtm_pos[k] = pos
    mtm_neg[k] = neg

#divide
EFV_pos = {}
EFV_neg = {}
for k,mtms in results_EFV.items():
    pos = {}
    neg = {}
    for w,mtm in mtms.items():
        tmp = mtm.copy()
        tmp[tmp<0] = 0
        pos[w] = tmp.mean(axis=1)

        tmp = mtm.copy()
        tmp[tmp>0] = 0
        neg[w] = tmp.mean(axis=1)
    EFV_pos[k] = pos
    EFV_neg[k] = neg
    
eval_y = 10
mtm_pos[eval_y]


#PERFIL CAPITAL
#Casos: Cash, Centrales o Nada
# APR 100%
apr = 1
k = {}
for y,mtms in results.items():
    k_cash, k_central,k_nada  = {}, {}, {}
    for prod,mtm in mtms.items():
        T = mtm.shape[0]
        cash = np.zeros_like(mtm)                
        central = np.zeros_like(mtm)                
        nada = np.zeros_like(mtm)                
        ##Colateral Cash
        if prod != 'BASIS_MTM' and prod !='CROSS_LIBOR_MTM':
            tmp = mtm[T-(DIAS_POR_ANO+1):T,:]
            cash[T-(DIAS_POR_ANO+1):T,:] = np.where(tmp>0,0,-apr*tmp)
            central[T-(DIAS_POR_ANO+1):T,:] = np.where(tmp>0,0,0)            
            nada[T-(DIAS_POR_ANO+1):T,:] = np.where(tmp>0,apr*tmp,0)
            if y>1 and y<=5:    
                tmp = mtm[0:T-(DIAS_POR_ANO+1),:]
                cash[0:T-(DIAS_POR_ANO+1)] = np.where(tmp>0,0.5/100,-apr*tmp+0.5/100*0.4)
                central[0:T-(DIAS_POR_ANO+1)] = np.where(tmp>0,0.5/100, 0.5/100*0.4)            
                nada[0:T-(DIAS_POR_ANO+1)] = np.where(tmp>0,apr*tmp+0.5/100, 0.5/100*0.4)            
            elif y>5:  
                tmp = mtm[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:]
                cash[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:] = np.where(tmp>0,0.5/100,-apr*tmp+0.5/100*0.4)    
                central[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:] = np.where(tmp>0,0.5/100,0.5/100*0.4)    
                nada[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:] = np.where(tmp>0,apr*tmp+0.5/100,0.5/100*0.4)    
                
                tmp = mtm[0:T-(DIAS_POR_ANO*5+1)]
                cash[0:T-(DIAS_POR_ANO*5+1),:] = np.where(tmp>0,1.5/100,-apr*tmp+0.015*0.4)    
                central[0:T-(DIAS_POR_ANO*5+1),:] = np.where(tmp>0,0.015,0.015*0.4)    
                nada[0:T-(DIAS_POR_ANO*5+1),:] = np.where(tmp>0,apr*tmp+1.5/100,0.015*0.4)    
        else:
            tmp = mtm[T-(DIAS_POR_ANO+1):T,:]
            cash[T-(DIAS_POR_ANO+1):T,:] = np.where(tmp>0,0.015,-apr*tmp+0.015*0.4)
            central[T-(DIAS_POR_ANO+1):T,:] = np.where(tmp>0,0.015,0.015*0.4)
            nada[T-(DIAS_POR_ANO+1):T,:] = np.where(tmp>0,apr*tmp+0.015,0.015*0.4)
            if y>1 and y<=5:
                tmp = mtm[0:T-(DIAS_POR_ANO+1),:]
                cash[0:T-(DIAS_POR_ANO+1)] = np.where(tmp>0,0.07,-apr*tmp+0.07*0.4)            
                central[0:T-(DIAS_POR_ANO+1)] = np.where(tmp>0,0.07,0.07*0.4)            
                nada[0:T-(DIAS_POR_ANO+1)] = np.where(tmp>0,apr*tmp+0.07,0.07*0.4)            
            elif y>5:  
                tmp = mtm[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:]                
                cash[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:] = np.where(tmp>0,0.07,-apr*tmp+0.07*0.4)    
                central[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:] = np.where(tmp>0,0.07,0.07*0.4)    
                nada[T-(DIAS_POR_ANO*5+1):T-(DIAS_POR_ANO+1),:] = np.where(tmp>0,apr*tmp+0.07,0.07*0.4)    
                
                tmp = mtm[0:T-(DIAS_POR_ANO*5+1)]
                cash[0:T-(DIAS_POR_ANO*5+1),:] = np.where(tmp>0,0.13,-apr*tmp+0.13*0.4)    
                central[0:T-(DIAS_POR_ANO*5+1),:] = np.where(tmp>0,0.13,0.13*0.4)    
                nada[0:T-(DIAS_POR_ANO*5+1),:] = np.where(tmp>0,apr*tmp+0.13,0.13*0.4)
        
        #   
        m = cash
        n = central  
        o = nada
        k_cash[prod] = np.mean(m,axis=1)
        k_central[prod] = np.mean(n,axis=1)
        k_nada[prod] = np.mean(o,axis=1)
        
    k[y] = {'cash':k_cash,'central':k_central,'nada':k_nada}
    
#Parametros
params = xw.sheets['params'].range('Q5:R11').options(pd.DataFrame,header=False).value
dur = xw.sheets['params'].range('T3:X7').options(pd.DataFrame).value

if side==1:
    first_row = 2
else:
    first_row = 8

#KVA
export = True
if export == True:
    s_k = params.loc['Spread Capital'][0]
    i = 0
    for y in k.keys():        
        w = pd.DataFrame(columns=['IRS_MTM','CROSS_MTM','BASIS_MTM','CROSS_LIBOR_MTM'])
        for gtype in k[y].keys():                        
            tmp = pd.DataFrame(k[y][gtype])                                                
            w.loc[gtype] = (tmp.sum().values*DT*s_k*10000)/dur[y].T.values
        xw.sheets['resultados-tabla'].range((first_row,2+5*i)).value = w
        i += 1
        
#CVA
if export == True:
    dP = xw.sheets['params'].range('J2:J5003').options(pd.DataFrame,index=False).value
    recovery = params.loc['Recuperacion'][0]
    w = pd.DataFrame(columns=['IRS_MTM','CROSS_MTM','BASIS_MTM','CROSS_LIBOR_MTM'])                     
    for y in mtm_pos.keys():                
        tmp = pd.DataFrame(mtm_pos[y]) 
        T = tmp.size 
        w.loc[y] = ((1-recovery)*(tmp.values.T @ dP[0:tmp.shape[0]].values)*10000).reshape(4)/dur[y].values
    xw.sheets['resultados-tabla'].range((first_row+12,2)).value = w
    
#FVA
if export == True:
    spread_cf = xw.sheets['params'].range('H2:H5003').options(pd.DataFrame,index=False).value
    w = pd.DataFrame(columns=['IRS_MTM','CROSS_MTM','BASIS_MTM','CROSS_LIBOR_MTM'])                     
    for y in EFV_pos.keys():                
        tmp = pd.DataFrame(EFV_pos[y]) + pd.DataFrame(EFV_neg[y]) 
        T = tmp.size 
        w.loc[y] = ((tmp.values.T @ spread_cf[0:tmp.shape[0]].values*DT)*10000).reshape(4)/dur[y].values
    xw.sheets['resultados-tabla'].range((first_row+12,8)).value = w

#COLVA
if export == True:
    spread_423 = xw.sheets['params'].range('E2:E5003').options(pd.DataFrame,index=False).value
    w = pd.DataFrame(columns=['IRS_MTM','CROSS_MTM','BASIS_MTM','CROSS_LIBOR_MTM'])                     
    for y in mtm_pos.keys():                
        tmp = pd.DataFrame(EFV_pos[y]) + pd.DataFrame(EFV_neg[y]) 
        w.loc[y] = ((tmp.values.T @ spread_423[0:tmp.shape[0]].values*DT)*10000).reshape(4)/dur[y].values
    xw.sheets['resultados-tabla'].range((first_row+12,14)).value = w