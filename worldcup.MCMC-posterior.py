# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:11:03 2022

@author: HZHONG
"""
## MCMC
import numpy as np
# import emcee
import pymc3 as pm
# 2022 goals data included worldcup
data=({'Germany':[2,1,1,1,1,5,0,3,1,1,1],
       'Japan':[2,2,2,1,4,0,4,0,6,0,3,2,0,1,2,0],
       'Costa Rica':[1,0,1,1,2,2,0,2,1,2,2,2,0,1],
       'Spain':[2,5,1,2,1,2,1,1,7,1],
       'United States':[1,0,3,0,5,0,3,5,1,0,0,1,0],
       'Iran':[1,1,0,2,1,1,1,1,2,2,2],
       'England':[2,3,0,1,0,0,0,3,6,0],
       'Wales':[2,1,1,1,1,1,2,1,0,1,0],
       'France':[2,5,1,1,1,0,2,0,4,2],
       'Tunisia':[0,4,0,1,0,1,0,4,0,2,3,1,1,2,0,0],
       'Denmark':[2,3,2,2,0,2,1,2,0,1],
       'Australia':[4,2,0,0,2,2,0,1,2,1,1],
       'Argentina':[2,1,3,1,3,5,3,3,5,1,2],
       'Poland':[1,2,2,1,2,0,0,1,1,0,2],
       'Mexico':[2,0,1,0,1,2,0,2,0,0,3,1,0,1,2,4,1,0,0],
       'Saudi Arabia':[1,0,1,1,0,0,0,0,0,1,1,0,2,0],
       'Portugal':[3,2,1,4,2,0,4,0,4,3,2],
       'South Korea':[5,4,1,2,2,0,1,2,2,4,3,3,0,2,1,1,0,2],
       'Croatia':[1,2,0,1,1,1,2,3,1,0,4],
       'Belgium':[2,3,1,6,1,1,2,0,1,1,0],
       'Canada':[2,2,2,0,4,0,4,1,2,0,2,2,0,1],
       'Morocco':[1,2,2,2,1,1,4,0,2,2,2,0,3,0,2],
       'Ghana':[0,0,1,2,0,1,3,1,1,0,3,1,2,2,0,1,2,2,3],
       'Uruguay':[1,4,1,2,3,0,2,0,0],
       'Cameroon':[2,4,1,2,2,0,3,0,2,1,0,2,0,0,1,1,0,3],
       'Brazil':[1,4,4,4,5,1,3,5,2,1],
       'Serbia':[1,0,0,4,1,2,4,2,5,0,3],
       'Switzerland':[1,1,1,0,0,1,2,2,0,1,0]})

def worldcup_posterior(a,b):
    # Poisson distribution for football games
    import numpy as np
    # import emcee
    import pymc3 as pm
    # import array as array
    
    # generate poisson data
    # pm.Poisson.dist(3).random(size=3)
    
    # team
    a=a
    b=b
    
    # a='Japan'
    # b='Spain'
    
    # Data before worldcup
    # data=({'Germany':[2,1,1,1,1,5,0,3,1],'Japan':[2,2,2,1,4,0,4,0,6,0,3,2,0,1],'Costa Rica':[1,0,1,1,2,2,0,2,1,2,2,2],'Spain':[2,5,1,2,1,2,1,1]})
    # team_a=np.array([2,1,1,1,1,5,0,3,1])
    
    team_a=data[a]
    team_b=data[b]
    # team_b=np.array([2,2,2,1,4,0,4,0,6,0,3,2,0,1])
    
    with pm.Model() as model:
        #prior
        # pm.HalfNormal.dist(sd=3).random(size=3)
        mu_a=pm.HalfNormal('mu_a',sd=3)
        mu_b=pm.HalfNormal('mu_b',sd=3)
    
    
        #posterior
        obs_a=pm.Poisson('obs_a',mu=mu_a,observed=team_a)
        obs_b=pm.Poisson('obs_b',mu=mu_b,observed=team_b)
        
        # set metroplis method
        # step=pm.Metropolis()
        
        step=pm.NUTS()
        # set times of sampling
        trace=pm.sample(20000,step=step)
        # set burned value
        burned_trace=trace[1000:]
        
    # with model:
    #     a_goals = trace['obs_a']
    
    with model:
        post_pred = pm.sample_posterior_predictive(burned_trace)
    #     prior_pred= pm.sample_prior_predictive(burned_trace)
        # mu_a_samples=burned_trace['mu_a']
    # np.histogram(obs_a)
        
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import poisson
    import pandas as pd
    
    # mu_a=burned_trace['mu_a']
    # mu_b=burned_trace['mu_b']
    
    
    t_a=post_pred['obs_a'].ravel()
    t_b=post_pred['obs_b'].ravel()
    
    a_goals,a_counts=np.unique(t_a,return_counts=True)
    a_prb=a_counts/sum(a_counts)
    # plt.bar(a_goals, a_counts)
    # plt.title('Team A')
    
    # plt.bar(a_goals,a_prb)
    # plt.title('Team A')
    
    
    # n,bins,patches=plt.hist(t_a,density=True,bins=range(14))
    # bin_centers=0.5*(bins[1:]+bins[:-1])
    # plt.plot(bin_centers,n,linewidth=3)
    
    # sns.kdeplot(t_a)
    # sns.histplot(t_a,kde=True,bins=range(13))
    # plt.show()
    
    b_goals,b_counts=np.unique(t_b,return_counts=True)
    b_prb=b_counts/sum(b_counts)
    # plt.bar(b_goals, b_counts)
    # plt.title('Team B')
    # plt.bar(b_goals,b_prb)
    # plt.title('Team B')
    # counts,bins=np.histogram(team_a)
    # plt.hist(x=team_a, bins=12,align='mid',color='#0504aa',rwidth=0.8)
    # plt.hist(bins[-1], bins, weights=counts)
    
    
    
    
    k=np.arange(7)
    
    # pmf_a=poisson.pmf(k,mu=mu_a.mean())
    # pmf_b=poisson.pmf(k,mu=mu_b.mean())
    
    df=pd.DataFrame(data={'Goals':list(k),'Prb_a':list(a_prb[0:7]),'Prb_b':list(b_prb[0:7])})
    df2=pd.DataFrame(data={'Goals':['>6'],'Prb_a':1-sum(a_prb[0:7]).tolist(),'Prb_b':1-sum(b_prb[0:7]).tolist()})
    df = df.append(df2)
    
    total_prb=pd.DataFrame()
    i=-1
    for x in df['Prb_a']:
            i+=1
            # print(i)
            total_prb['A Goals-{}'.format(i)]=x*df['Prb_b']
            # total_prb.append(x*df['Prb_b'])
            
    # calculate win, lose and draw probability
    prb_a_win=np.triu(total_prb,1).sum()
    prb_b_win=np.tril(total_prb,-1).sum()
    prb_draw=np.diagonal(total_prb).sum()
    
    total_prb.columns=df['Goals']
    total_prb.index=df['Goals']
    
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(16,14))
    fmt = lambda x,pos: '{:.1%}'.format(x)
    ax=sns.heatmap(total_prb, annot=True,fmt='.2%',annot_kws={"fontsize":12},linewidths=0.5,cbar_kws={'format':FuncFormatter(fmt)})
    ax.invert_yaxis()
    plt.title(str(a)+' VS '+str(b)+"  "+str(a)+' Win: {}%, Lose: {}%, Draw: {}%'.format(round(prb_a_win*100,2),round(prb_b_win*100,2),round(prb_draw*100,2)),fontsize=15)
    plt.xlabel(str(a)+" Goals",fontsize=12)
    plt.ylabel(str(b)+" Goals",fontsize=12)
    figure=ax.get_figure()
    figure.set_size_inches(16,14)
    figure.savefig('C:/Users/HZHONG/Documents/Python Scripts/football/'+str(a)+' VS '+str(b)+'-Posterior.png', dpi=200)


# worldcup_posterior('Ghana','Uruguay')
# worldcup_posterior('Germany','Costa Rica')
# worldcup_posterior('Serbia','Switzerland')
# worldcup_posterior('South Korea','Portugal')
# worldcup_posterior('Brazil','Cameroon')
# worldcup_posterior('Croatia','Belgium')
# worldcup_posterior('Canada','Morocco')
# worldcup_posterior('Japan','Spain')



# Updated model

# def worldcup_posterior(a,b):
#     # Poisson distribution for football games
#     import numpy as np
#     # import emcee
#     import pymc3 as pm
#     # import array as array
    
#     # generate poisson data
#     # pm.Poisson.dist(3).random(size=3)
    
#     # team
#     a=a
#     b=b
    
#     # a='Japan'
#     # b='Spain'
    
#     # Data before worldcup
#     # data=({'Germany':[2,1,1,1,1,5,0,3,1],'Japan':[2,2,2,1,4,0,4,0,6,0,3,2,0,1],'Costa Rica':[1,0,1,1,2,2,0,2,1,2,2,2],'Spain':[2,5,1,2,1,2,1,1]})
#     # team_a=np.array([2,1,1,1,1,5,0,3,1])
    
#     team_a=data[a]
#     team_b=data[b]
#     # team_b=np.array([2,2,2,1,4,0,4,0,6,0,3,2,0,1])
# his_a=({'England':[2,3,0,1,0,0,0,3,6,0]})
# his_b=({'France':[2,5,1,1,1,0,2,0,4,2]})

# n_a=({'England':[6,0,3,3]})
# n_b=({'France':[4,2,0,3]})

# C_a=({'England':[0.78,0.54,0.65,0.59]})
# A_a=({'England':[0.402597403,0.481132075,0.452173913,0.380165289]})
# S_a=({'England':[0.538461538,0.428571429,0.388888889,0.571428571]})


# C_b=({'France':[0.538461538,0.428571429,0.388888889,0.571428571]})

# A_b=({'France':[0.473684211,0.453703704,0.369127517,0.623853211]})

# S_b=({'France':[0.4375,0.3,0.3,0.470588235]})

history=({'England':[2,3,0,1,0,0,0,3],
          'France':[2,5,1,1,1,0,2,0],
          'Argentina':[2,1,3,1,3,5,3,3,5,1,2],
          'Netherlands':[4,1,4,2,2,3,2,1],
          'Brazil':[1,4,4,4,5,1,3,5],
          'Croatia':[1,2,0,1,1,1,2,3,1],
          'Spain':[2,5,1,2,1,2,1,1],
          'Portugal':[3,2,1,4,2,0,4,0,4],
          'Morogo':[1,2,2,2,1,1,4,0,2,2,2,0,3]
          
         })

now_goals=({'England':[6,0,3,3],
      'France':[4,2,0,3,2,2],
      'Argentina':[1,2,2,2,2,3],
      'Netherlands':[2,1,2,3],
      'Brazil':[2,1,0,4],
      'Croatia':[0,4,0,1,1,0],
      'Spain':[7,1,1],
      'Portugal':[3,2,1,6],
      'Morogo':[0,2,2,0,1,0]
      
      })

control_rate=({'England':[0.78,0.54,0.65,0.59],
                'Netherlands':[0.53,0.48,0.61,0.42],
                'Argentina':[0.66,0.55,0.72,0.61,0.52,0.42],
                'France':[0.62,0.51,0.62,0.49,0.42,0.4],
                'Brazil':[0.62,0.59,0.65,0.51],
                'Croatia':[0.61,0.4,0.43,0.58,0.58,0.58],
                'Spain':[0.76,0.56,0.82],
                'Portugal':[0.62,0.55,0.58,0.45],
                'Morogo':[0.39,0.67,0.43,0.26,0.24,0.6]
                })
                
attack_rate=({'England':[0.402597403,0.481132075,0.452173913,0.380165289],
              'Netherlands':[0.546391753,0.27,0.37398374,0.425],
              'Argentina':[0.6,0.368421053,0.523529412,0.320895522,0.276315789,0.2372881],
              'France':[0.473684211,0.453703704,0.369127517,0.623853211,0.382352941,0.1900826],
              'Brazil':[0.454545455,0.465753425,0.697368421,0.474137931],
              'Croatia':[0.28030303,0.6,0.623853211,0.377906977,0.387096774,0.2682927],
              'Spain':[0.346534653,0.383838384,0.303664921,0.2682927],
              'Portugal':[0.475806452,0.423423423,0.57,0.268518],
              'Morogo':[0.29661,0.264367,0.271739,0.3,0.404761905,0.2941176]
              })


              
shoot_rate=({'England':[0.538461538,0.428571429,0.388888889,0.571428571],
             'Netherlands':[0.3,0.5,0.333333333,0.6],
             'Argentina':[0.545454545,0.5,0.541666667,0.416666667,0.4,0.7],
             'France':[0.4375,0.3,0.3,0.470588235,0.625,0.2],
             'Brazil':[0.458333333,0.555555556,0.333333333,0.555555556],
             'Croatia':[0.333333333,0.769230769,0.363636364,0.235294118,0.111111111,0.1666667],
             'Spain':[0.411764706,0.428571429,0.384615385],
             'Portugal':[0.454545455,0.266666667,0.416666667,0.181818],
             'Morogo':[0.25,0.4,0.4,0.5,0.333333333,0.2307692]
             })


# # weight
# w_his=0.25
# w_now=0.75
# import numpy as np
# # import emcee
# import pymc as pm


def worldcup_posterior(a,b,w_now=0.75,a_confidence=1,b_confidence=1): 
    import numpy as np
    # import emcee
    import pymc3 as pm
    w_his=1-w_now
    a=a
    b=b
    his_a=history[a]
    his_b=history[b]
    n_a=now_goals[a]
    n_b=now_goals[b]
    C_a=control_rate[a]
    C_b=control_rate[b]
    A_a=attack_rate[a]
    A_b=attack_rate[b]
    S_a=shoot_rate[a]
    S_b=shoot_rate[b]
    
    
    with pm.Model() as model:
        #prior
        # pm.HalfNormal.dist(sd=3).random(size=3)
        # mu_a=pm.HalfNormal('mu_a',sd=3)
        # mu_b=pm.HalfNormal('mu_b',sd=3)
        mu_a_h=pm.HalfNormal('mu_a_h',sigma=3)
        mu_b_h=pm.HalfNormal('mu_b_h',sigma=3)
        # sigma_a=pm.HalfCauchy('sigma_a',1)
        # sigma_b=pm.HalfCauchy('sigma_b',1)
        lamda_a_h=pm.Poisson('lambda_a_h',mu=mu_a_h, observed=his_a)
        lamda_b_h=pm.Poisson('lambda_b_h',mu=mu_b_h, observed=his_b)
        
        k_a_1=pm.HalfNormal('k_a_1',sigma=2)
        k_a_2=pm.HalfNormal('k_a_2',sigma=2)
        k_a_3=pm.HalfNormal('k_a_3',sigma=2)
        
        k_b_1=pm.HalfNormal('k_b_1',sigma=2)
        k_b_2=pm.HalfNormal('k_b_2',sigma=2)
        k_b_3=pm.HalfNormal('k_b_3',sigma=2)
        
        sigma_mu_a_n=pm.HalfCauchy('sigma_mu_a_n',1)
        sigma_mu_b_n=pm.HalfCauchy('sigma_mu_b_n',1)
        
        intercept_a = pm.Normal("intercept_a", 0, sigma=20)
        intercept_b = pm.Normal("intercept_b", 0, sigma=20)
        
        # C_a=pm.Uniform('C_a',0,1)
        # A_a=pm.Uniform('A_a',0,1)
        # S_a=pm.Uniform('S_a',0,1)
        # C_b=pm.Uniform('C_b',0,1)
        # A_b=pm.Uniform('A_b',0,1)
        # S_b=pm.Uniform('S_b',0,1)
        
        # obs_C_a=pm.Uniform('obs_C_a',0,1)
        obs_mu_a_n=pm.Normal('mu_a_n',mu=k_a_1*C_a+k_a_2*A_a+k_a_3*S_a+intercept_a,sigma=sigma_mu_a_n,observed=n_a)
        obs_mu_b_n=pm.Normal('mu_b_n',mu=k_b_1*C_b+k_b_2*A_b+k_b_3*S_b+intercept_b,sigma=sigma_mu_b_n,observed=n_b)
        
        lambda_a=pm.Deterministic('lambda_a',(((mu_a_h*1)+(k_a_1*C_a+k_a_2*A_a+k_a_3*S_a+intercept_a)*(w_now/w_his))/(1+w_now/w_his)))
        lambda_b=pm.Deterministic('lambda_b',(((mu_b_h*1)+(k_b_1*C_a+k_b_2*A_b+k_b_3*S_b+intercept_b)*(w_now/w_his))/(1+w_now/w_his)))
    
                                  
        # lambda_a=((mu_a_h)*1+(k_a_1*C_a+k_a_2*A_a+k_a_3*S_a+intercept_a)*(w_now/w_his))/(1+w_now/w_his)
        # lambda_b=((mu_b_h)*1+(k_b_1*C_a+k_b_2*A_b+k_b_3*S_b+intercept_b)*(w_now/w_his))/(1+w_now/w_his)
        #posterior
        obs_a=pm.Poisson('obs_a',mu=(lambda_a)*a_confidence,observed=n_a)
        obs_b=pm.Poisson('obs_b',mu=(lambda_b)*b_confidence,observed=n_b)
        
        
        # set metroplis method
        # step=pm.Metropolis()
        
        step=pm.NUTS()
        # step=pm.HamiltonianMC()
        # set times of sampling
        trace=pm.sample(30000,tune=10000,step=step)
        # set burned value
        burned_trace=trace[2000:]
        
    # with model:
    #     a_goals = trace['obs_a']
    
    with model:
        post_pred = pm.sample_posterior_predictive(burned_trace)


    t_a=post_pred['obs_a'].ravel()
    t_b=post_pred['obs_b'].ravel()
    
    a_goals,a_counts=np.unique(t_a,return_counts=True)
    a_prb=a_counts/sum(a_counts)
 
    b_goals,b_counts=np.unique(t_b,return_counts=True)
    b_prb=b_counts/sum(b_counts)
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    # from scipy.stats import poisson
    import pandas as pd
    k=np.arange(7)
    
    # pmf_a=poisson.pmf(k,mu=mu_a.mean())
    # pmf_b=poisson.pmf(k,mu=mu_b.mean())
    
    df=pd.DataFrame(data={'Goals':list(k),'Prb_a':list(a_prb[0:7]),'Prb_b':list(b_prb[0:7])})
    df2=pd.DataFrame(data={'Goals':['>6'],'Prb_a':1-sum(a_prb[0:7]).tolist(),'Prb_b':1-sum(b_prb[0:7]).tolist()})
    df = df.append(df2)
    
    total_prb=pd.DataFrame()
    i=-1
    for x in df['Prb_a']:
            i+=1
            # print(i)
            total_prb['A Goals-{}'.format(i)]=x*df['Prb_b']
            # total_prb.append(x*df['Prb_b'])
            
    # calculate win, lose and draw probability
    prb_a_win=np.triu(total_prb,1).sum()
    prb_b_win=np.tril(total_prb,-1).sum()
    prb_draw=np.diagonal(total_prb).sum()
    
    total_prb.columns=df['Goals']
    total_prb.index=df['Goals']
    
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(16,14))
    fmt = lambda x,pos: '{:.1%}'.format(x)
    ax=sns.heatmap(total_prb, annot=True,fmt='.2%',annot_kws={"fontsize":12},linewidths=0.5,cbar_kws={'format':FuncFormatter(fmt)})
    ax.invert_yaxis()
    plt.title(str(a)+' '+str(a_confidence*100)+'% Performance'+' VS '+str(b)+' '+str(b_confidence*100)+'% Performance'+"  "+str(a)+' Win: {}%, Lose: {}%, Draw: {}%'.format(round(prb_a_win*100,2),round(prb_b_win*100,2),round(prb_draw*100,2)),fontsize=15)
    plt.xlabel(str(a)+" Goals",fontsize=12)
    plt.ylabel(str(b)+" Goals",fontsize=12)
    figure=ax.get_figure()
    figure.set_size_inches(16,14)
    figure.savefig('C:/Users/HZHONG/Documents/Python Scripts/football/'+str(a)+' VS '+str(b)+' '+str(w_his*100)+'% history accounted'+'-Posterior.png', dpi=200)