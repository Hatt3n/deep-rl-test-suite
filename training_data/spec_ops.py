"""


Last edit: 2022-06-27
"""

"""Unique configurations:"""
# Cartpole:
# [64_relu, 256_128_relu] for [a2c_s]
# [64_relu, 64_64_relu, 64_64_tanh, 400_300_relu] for [dqn]
# [64_relu, 64_64_relu] for [reinforce]
# [256_128_relu] for [a2c_s_adam]
# ===> Proposed experiments ===>:
# [64_relu, 64_64_relu, 64_64_tanh, 256_128_relu, 400_300_relu] for [a2c_s, dqn, reinforce]
"""
{'cartpole': 
    {'64_relu': {
        'a2c_s': array([200., 200., 104., 123., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]), 
        'dqn': array([200., 200.,  98., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]), 
        'reinforce': array([200., 192., 200., 200., 198., 200., 200., 200., 200., 200., 200., 200., 200., 200., 197., 200., 200., 187., 200., 200.])}, 
    '64_64_relu': {
        'a2c_s': array([200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]), 
        'dqn': array([200., 200., 200., 200., 200., 200.,  25.,  26.,  33.,  28., 148., 140., 139., 152., 147., 120., 111., 121., 124., 120.]), 
        'reinforce': array([ 20.,  28., 107.,  26.,  13., 200., 186., 197., 198., 167., 200., 200., 200., 191., 200., 180., 172., 176., 200., 197.])}, 
    '64_64_tanh': {
        'a2c_s': array([200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 104., 159., 118., 121., 113.]), 
       'dqn': array([ 91.,  93., 200.,  98., 155.,  20.,  21.,  22.,  23.,  18., 122., 200., 182., 200., 200., 200.,  78., 101., 138., 195.]), 
       'reinforce': array([200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.])}, 
    '256_128_relu': {
        'a2c_s': array([ 9., 10., 10.,  8., 10.,  8.,  9.,  8.,  9., 10., 10.,  9.,  8., 11., 10.,  9., 11.,  9.,  9.,  8.]), 
        'dqn': array([200., 200., 200., 200., 200., 200., 200., 194., 200., 200., 182., 171., 168., 186., 182., 200., 188., 200., 200., 200.]), 
        'reinforce': array([200., 200., 200., 200., 200., 115., 121., 109., 116., 149., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.])}, 
    '400_300_relu': {
        'a2c_s': array([ 9., 10., 10.,  8., 10.,  8.,  9.,  8.,  9., 10., 10.,  9.,  8., 11., 10.,  9., 11.,  9.,  9.,  8.]), 
        'dqn': array([128., 129., 132., 125., 130., 179., 177., 172., 162., 171.,  14., 12.,  11.,  12.,  14., 200., 200., 200., 200., 200.]), 
        'reinforce': array([200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 110., 104., 101., 118.,  71., 200., 200., 200., 200., 200.])}}, 
'_debug': 'eval_table created at 2022-06-22 20:19:25.705112, using seeds [0, 10, 100, 1000].'}
"""
# [256_128_relu] for [a2c_s_adam]
"""
{'cartpole': {
    '256_128_relu': {
        'a2c_s_adam': array([ 54.,  72.,  32.,  43., 134.,  14.,  86.,  59.,  54.,  20., 140., 49., 158., 200.,  15.,  10.,  59., 200.,  64.,  56.])}}, 
'_debug': 'eval_table created at 2022-06-22 20:23:46.631649, using seeds [0, 10, 100, 1000].'}
"""

# Furuta paper:
# [64_relu, 64_64_relu, 64_64_tanh] for [reinforce, a2c_s]
# ===> Proposed experiments ===>:
# [64_relu, 64_64_relu, 64_64_tanh] for [reinforce, a2c_s]
"""
{'furuta_paper': {
    '64_relu': {
        'a2c_s': array([0.02, 0.02, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]), 
       'reinforce': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}, 
    '64_64_relu': {
        'a2c_s': array([1.44, 1.36, 1.6 , 1.52, 1.46, 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.02, 0.02, 0.02, 0.02]), 
       'reinforce': array([1.46, 1.44, 1.34, 1.62, 1.38, 1.14, 1.74, 1.84, 1.64, 1.26, 1.52, 1.6 , 1.58, 1.8 , 1.46])}, 
    '64_64_tanh': {
        'a2c_s': array([3.42, 1.12, 1.94, 3.36, 2.06, 0.  , 0.  , 0.58, 0.  , 0.46, 0.12, 0.12, 0.22, 0.12, 0.  ]), 
       'reinforce': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}}, 
'_debug': 'eval_table created at 2022-06-22 20:33:25.297732, using seeds [0, 10, 1000].'}
"""

# PBRS 1:
# [64_64_relu, 256_128_relu] for [ddpg]
# [64_64_relu] for [reinforce]
# ===> Proposed experiments ===>:
# [64_64_relu, 256_128_relu] for [ddpg, reinforce]
"""
{'furuta_pbrs': {
    '64_64_relu': {
        'ddpg': array([9.26, 9.26, 9.26, 9.26, 9.26, 1.38, 1.38, 1.38, 1.38, 1.38, 7.16, 7.16, 7.16, 7.16, 7.16, 4.48, 4.48, 4.48, 4.48, 4.48]), 
       'reinforce': array([0.  , 0.  , 0.  , 0.  , 0.  , 1.08, 1.68, 2.32, 1.04, 0.94, 1.94, 2.56, 1.88, 1.66, 1.82, 0.8 , 1.72, 0.78, 1.08, 3.28])}, 
    '256_128_relu': {
        'ddpg': array([4.  , 4.  , 4.  , 4.  , 4.  , 9.48, 9.48, 9.48, 9.48, 9.48, 5.84, 5.84, 5.84, 5.84, 5.84, 4.68, 4.68, 4.68, 4.68, 4.68]), 
       'reinforce': array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.62, 0.8 , 1.06, 0.9 , 0.82, 0.  , 0.  , 0.  , 0.  , 0.  ])}}, 
'_debug': 'eval_table created at 2022-06-22 20:45:29.012152, using seeds [0, 10, 100, 1000].'}
"""

# PBRS 2:
# [64_64_relu] for [ddpg, reinforce]
# ===> Proposed experiments ===>:
# [64_64_relu] for [ddpg, reinforce]
"""
{'furuta_pbrs2': {
    '64_64_relu': {
        'ddpg': array([9.18, 9.18, 9.18, 9.18, 9.18, 5.12, 5.12, 5.12, 5.12, 5.12, 9.28, 9.28, 9.28, 9.28, 9.28, 6.26, 6.26, 6.26, 6.26, 6.26]), 
       'reinforce': array([1.6 , 1.28, 1.64, 3.94, 2.04, 0.  , 0.  , 0.  , 0.  , 0.  , 0.1 , 0.22, 0.24, 0.  , 0.  , 2.46, 2.18, 2.76, 1.96, 3.8 ])}}, 
'_debug': 'eval_table created at 2022-06-22 20:47:06.498066, using seeds [0, 10, 100, 1000].'}
"""

# PBRS 3:
# [64_64_tanh, 64_64_relu, 256_128_relu] for [ddpg]
# [64_64_relu] for [reinforce]
# [64_64_tanh] for [ppo]
# [64_64_relu, 256_128_relu] for [ppo_s]
# ===> Proposed experiments ===>:
# [64_64_tanh, 64_64_relu, 256_128_relu] for [ddpg, reinforce, ppo, ppo_s]
"""
{'furuta_pbrs3': {
    '64_64_relu': {
        'ddpg': array([9.18, 9.18, 9.18, 9.18, 9.18, 9.36, 9.36, 9.36, 9.36, 9.36, 5.94, 5.94, 5.94, 5.94, 5.94, 8.34, 8.34, 8.34, 8.34, 8.34]), 
       'ppo_s': array([1.26, 1.16, 2.6 , 1.16, 1.32, 8.16, 6.98, 4.76, 7.48, 8.24, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]), 
       'ppo': array([2.66, 2.86, 2.9 , 2.78, 3.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 4.56, 4.22, 4.  , 4.06, 3.94]), 
       'reinforce': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}, 
    '64_64_tanh': {
        'ddpg': array([8.88, 8.88, 8.88, 8.88, 8.88, 3.46, 3.46, 3.46, 3.46, 3.46, 8.66, 8.66, 8.66, 8.66, 8.66, 3.72, 3.72, 3.72, 3.72, 3.72]), 
       'ppo_s': array([1.52, 1.64, 1.58, 1.42, 1.68, 3.6 , 0.  , 3.94, 1.24, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.74, 0.68, 0.74, 0.7 , 0.82]), 
       'ppo': array([5.78, 4.6 , 5.4 , 3.9 , 4.98, 4.48, 4.68, 8.38, 8.38, 8.4 , 1.1 , 1.04, 0.98, 1.04, 1.04, 3.64, 7.26, 3.88, 5.06, 3.82]), 
       'reinforce': array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.54, 0.48, 0.64, 0.6 , 0.6 , 0.  , 0.  , 0.  , 0.  , 0.  ])}, 
    '256_128_relu': {
        'ddpg': array([8.88, 8.88, 8.88, 8.88, 8.88, 8.36, 8.36, 8.36, 8.36, 8.36, 9.2 , 9.2 , 9.2 , 9.2 , 9.2 , 8.64, 8.64, 8.64, 8.64, 8.64]), 
       'ppo_s': array([8.14, 8.14, 8.14, 8.14, 8.14, 5.58, 7.46, 6.5 , 5.7 , 6.8 , 6.8 , 5.5 , 4.28, 4.42, 8.16, 1.26, 2.66, 3.5 , 2.8 , 1.44]), 
       'ppo': array([1.52, 1.46, 1.44, 1.42, 1.48, 2.4 , 2.36, 2.4 , 2.24, 2.16, 1.14, 1.14, 1.14, 1.18, 1.14, 0.  , 0.  , 0.  , 0.  , 0.  ]), 
       'reinforce': array([0.  , 0.  , 0.  , 0.  , 0.  , 1.38, 1.3 , 1.22, 1.18, 1.44, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])}}, 
'_debug': 'eval_table created at 2022-06-22 20:52:26.241794, using seeds [0, 10, 100, 1000].'}
"""

# QUBE 2:
"""
{'qube2_sim': {
    '64_64_relu': {
        'ddpg': array([2.564, 4.112, 2.316, 5.36 , 3.992, 6.26 , 6.276, 6.268, 6.264, 6.248, 6.988, 5.808, 1.892, 6.988, 7.008, 0.532, 0.872, 0.376, 0.156, 0.696])}}, 
'_debug': 'eval_table created at 2022-06-22 22:46:22.098869, using seeds [0, 10, 100, 1000].'}
{'qube2_sim': {
    '64_64_relu': {
        'a2c': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
       'ppo': array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.452, 0.036, 0.   , 0.24 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])}}, 
'_debug': 'eval_table created at 2022-06-22 22:49:23.903432, using seeds [0, 10, 100, 1000].'}
"""

# QUBE 2 REAL SEED 0:
"""
Independently defined evaluation data: 
[3.792000000000003, 4.159999999999985, 5.38399999999985, 2.836000000000002, 3.856000000000003, 3.796000000000003, 4.243999999999976, 5.123999999999879, 5.139999999999877, 4.167999999999984, 2.876000000000002, 5.455999999999842, 5.38399999999985, 0.19600000000000015, 3.636000000000003, 4.447999999999953, 5.147999999999876, 1.5120000000000011, 2.3280000000000016, 5.179999999999873]
------------------------------------
----> Results of evaluation: mean 3.9331999999999487, std 1.37407196318092, across 20 tests. <----
------------------------------------
"""

# QUBE 2 REAL ALL SEEDS:
"""
------------------------------------
----> Results of evaluation: mean 1.0953999999999915, std 1.517891840679015, across 20 tests. <----
------------------------------------
[2.376, 3.272, 2.048, 5.384, 4.26, 0.34, 0.188, 0.204, 0.496, 0.572, 0.096, 1.148, 0.148, 0.268, 0.12, 0.192, 0.188, 0.216, 0.212, 0.18]
"""

# QUBE 2 SIM SEED 0:
"""
{'qube2_sim': {
    '64_64_relu': {
        'ddpg': [1.9360000000000015, 0.7760000000000006, 3.856000000000003, 3.760000000000003, 4.451999999999953, 5.8079999999998035, 0.10800000000000007, 3.972000000000003, 5.8079999999998035, 5.8439999999998, 2.988000000000002, 2.2080000000000015, 1.0440000000000007, 0.20800000000000016, 3.5640000000000027, 2.672000000000002, 1.5400000000000011, 5.8439999999998, 5.851999999999799, 2.3320000000000016]}}, 
'_debug': 'eval_table created at 2022-06-27 12:37:54.867968, using seeds [0].'}
"""

"""The tests:"""
# These tests use https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ranksums.html

# 0. Cartpole: {64} ReLU and {64, 64} ReLU for DQN
# DQN 64_relu [200., 200.,  98., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]
# DQN 64_64_relu [200., 200., 200., 200., 200., 200.,  25.,  26.,  33.,  28., 148., 140., 139., 152., 147., 120., 111., 121., 124., 120.]
# ===> RanksumsResult(statistic=3.435361308082917, pvalue=0.0005917642796747578)


# 1. Cartpole: {64} ReLU and {64, 64} ReLU for REINFORCE
# reinforce 64_relu [200., 192., 200., 200., 198., 200., 200., 200., 200., 200., 200., 200., 200., 200., 197., 200., 200., 187., 200., 200.]
# reinforce 64_64_relu [ 20.,  28., 107.,  26.,  13., 200., 186., 197., 198., 167., 200., 200., 200., 191., 200., 180., 172., 176., 200., 197.]
# ===> RanksumsResult(statistic=3.2054355512427217, pvalue=0.0013485824363168647)

# 2. Cartpole: {64} ReLU and {64, 64} tanh for DQN
# DQN 64_relu [200., 200.,  98., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]
# DQN 64_64_tanh [ 91.,  93., 200.,  98., 155.,  20.,  21.,  22.,  23.,  18., 122., 200., 182., 200., 200., 200.,  78., 101., 138., 195.]
# ===> RanksumsResult(statistic=3.8140625546432383, pvalue=0.00013670084197896473)


# 3. Cartpole: {64} ReLU and {256, 128} ReLU for A2C from SLM Lab
# A2C_s 64_relu [200., 200., 104., 123., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]
# A2C_s 256_128_relu [ 9., 10., 10.,  8., 10.,  8.,  9.,  8.,  9., 10., 10.,  9.,  8., 11., 10.,  9., 11.,  9.,  9.,  8.]
# ===> RanksumsResult(statistic=5.410017808004594, pvalue=6.301848221392269e-08)


# 4. Cartpole: {64} ReLU and {400, 300} ReLU for DQN
# DQN 64_relu [200., 200.,  98., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.]
# DQN 400_300_relu [128., 129., 132., 125., 130., 179., 177., 172., 162., 171.,  14., 12.,  11.,  12.,  14., 200., 200., 200., 200., 200.]
# ===> RanksumsResult(statistic=3.719387243003158, pvalue=0.0001997066750366024)


# 5. Cartpole: {256, 128} ReLU for A2C from SLM Lab, RMSProp vs RAdam
# A2C_s 256_128_relu [ 9., 10., 10.,  8., 10.,  8.,  9.,  8.,  9., 10., 10.,  9.,  8., 11., 10.,  9., 11.,  9.,  9.,  8.]
# A2C_s_adam 256_128_relu [ 54.,  72.,  32.,  43., 134.,  14.,  86.,  59.,  54.,  20., 140., 49., 158., 200.,  15.,  10.,  59., 200.,  64.,  56.]
# ===> RanksumsResult(statistic=-5.274767362804479, pvalue=1.329245985375027e-07)


# 6. Furuta paper: {64} ReLU, {64, 64} ReLU, and {64, 64} tanh for REINFORCE
# reinforce 64_relu and 64_64_tanh [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
# reinforce 64_64_relu [1.46, 1.44, 1.34, 1.62, 1.38, 1.14, 1.74, 1.84, 1.64, 1.26, 1.52, 1.6 , 1.58, 1.8 , 1.46]
# ===> RanksumsResult(statistic=-4.666282626286914, pvalue=3.0669777654622667e-06)


# 7. Furuta paper: {64, 64} ReLU for REINFORCE vs {64, 64} tanh A2C from SLM Lab
# reinforce 64_64_relu [1.46, 1.44, 1.34, 1.62, 1.38, 1.14, 1.74, 1.84, 1.64, 1.26, 1.52, 1.6 , 1.58, 1.8 , 1.46]
# A2C_s 64_64_tanh [3.42, 1.12, 1.94, 3.36, 2.06, 0.  , 0.  , 0.58, 0.  , 0.46, 0.12, 0.12, 0.22, 0.12, 0.  ]
# ===> RanksumsResult(statistic=2.177598558933893, pvalue=0.0294359369343824)


# 8. Furuta paper: {64} ReLU, {64, 64} ReLU for A2C from SLM Lab
# A2C_s 64_relu [0.02, 0.02, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
# A2C_s 64_64_relu [1.44, 1.36, 1.6 , 1.52, 1.46, 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.02, 0.02, 0.02, 0.02]
# ===> RanksumsResult(statistic=-2.6960744062991058, pvalue=0.007016199234239618)


# 9. Furuta paper: {64} ReLU, {64, 64} tanh for A2C from SLM Lab
# A2C_s 64_relu [0.02, 0.02, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
# A2C_s 64_64_tanh [3.42, 1.12, 1.94, 3.36, 2.06, 0.  , 0.  , 0.58, 0.  , 0.46, 0.12, 0.12, 0.22, 0.12, 0.  ]
# ===> RanksumsResult(statistic=-3.2560283214535355, pvalue=0.0011298248105825034)


# 10. PBRS 3: {64, 64} tanh for DDPG vs PPO
# DDPG 64_64_tanh [8.88, 8.88, 8.88, 8.88, 8.88, 3.46, 3.46, 3.46, 3.46, 3.46, 8.66, 8.66, 8.66, 8.66, 8.66, 3.72, 3.72, 3.72, 3.72, 3.72]
# PPO 64_64_tanh [5.78, 4.6 , 5.4 , 3.9 , 4.98, 4.48, 4.68, 8.38, 8.38, 8.4 , 1.1 , 1.04, 0.98, 1.04, 1.04, 3.64, 7.26, 3.88, 5.06, 3.82]
# ===> RanksumsResult(statistic=1.4877548972012633, pvalue=0.13681554418853384)


# 11. PBRS 1 and 3: {256, 128} ReLU for DDPG
# 1 DDPG 256_128_relu [4.  , 4.  , 4.  , 4.  , 4.  , 9.48, 9.48, 9.48, 9.48, 9.48, 5.84, 5.84, 5.84, 5.84, 5.84, 4.68, 4.68, 4.68, 4.68, 4.68]
# 3 DDPG 256_128_relu [8.88, 8.88, 8.88, 8.88, 8.88, 8.36, 8.36, 8.36, 8.36, 8.36, 9.2 , 9.2 , 9.2 , 9.2 , 9.2 , 8.64, 8.64, 8.64, 8.64, 8.64]
# ===> RanksumsResult(statistic=-2.705008904002297, pvalue=0.006830255863044082)


# 12. PBRS 1 and 3: {64, 64} ReLU for REINFORCE
# 1 reinforce 64_64_relu [0.  , 0.  , 0.  , 0.  , 0.  , 1.08, 1.68, 2.32, 1.04, 0.94, 1.94, 2.56, 1.88, 1.66, 1.82, 0.8 , 1.72, 0.78, 1.08, 3.28]
# 3 reinforce 64_64_relu [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
# ===> RanksumsResult(statistic=4.0575133560034455, pvalue=4.959798167765305e-05)


# 13. PBRS 2 and 3: {64, 64} ReLU for REINFORCE
# 2 reinforce 64_64_relu [1.6 , 1.28, 1.64, 3.94, 2.04, 0.  , 0.  , 0.  , 0.  , 0.  , 0.1 , 0.22, 0.24, 0.  , 0.  , 2.46, 2.18, 2.76, 1.96, 3.8 ]
# 3 reinforce 64_64_relu [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
# ===> RanksumsResult(statistic=3.5165115752029856, pvalue=0.0004372575927067331)


# 14. PBRS 1 and 3: {64, 64} ReLU for DDPG
# 1 DDPG 64_64_relu [9.26, 9.26, 9.26, 9.26, 9.26, 1.38, 1.38, 1.38, 1.38, 1.38, 7.16, 7.16, 7.16, 7.16, 7.16, 4.48, 4.48, 4.48, 4.48, 4.48]
# 3 DDPG 64_64_relu [9.18, 9.18, 9.18, 9.18, 9.18, 9.36, 9.36, 9.36, 9.36, 9.36, 5.94, 5.94, 5.94, 5.94, 5.94, 8.34, 8.34, 8.34, 8.34, 8.34]
# ===> RanksumsResult(statistic=-2.705008904002297, pvalue=0.006830255863044082)


# 15. PBRS 1 and 2: {64, 64} ReLU for DDPG
# 1 DDPG 64_64_relu [9.26, 9.26, 9.26, 9.26, 9.26, 1.38, 1.38, 1.38, 1.38, 1.38, 7.16, 7.16, 7.16, 7.16, 7.16, 4.48, 4.48, 4.48, 4.48, 4.48]
# 2 DDPG 64_64_relu [9.18, 9.18, 9.18, 9.18, 9.18, 5.12, 5.12, 5.12, 5.12, 5.12, 9.28, 9.28, 9.28, 9.28, 9.28, 6.26, 6.26, 6.26, 6.26, 6.26]
# ===> RanksumsResult(statistic=-2.0287566780017228, pvalue=0.04248307982571583)


# 16. PBRS 3: {64, 64} ReLU, {256, 128} ReLU for DDPG
# DDPG 64_64_relu [9.18, 9.18, 9.18, 9.18, 9.18, 9.36, 9.36, 9.36, 9.36, 9.36, 5.94, 5.94, 5.94, 5.94, 5.94, 8.34, 8.34, 8.34, 8.34, 8.34]
# DDPG 256_128_relu [8.88, 8.88, 8.88, 8.88, 8.88, 8.36, 8.36, 8.36, 8.36, 8.36, 9.2 , 9.2 , 9.2 , 9.2 , 9.2 , 8.64, 8.64, 8.64, 8.64, 8.64]
# ===> RanksumsResult(statistic=-0.6762522260005742, pvalue=0.4988805190741191)


# 17. PBRS 3: {64, 64} ReLU, {256, 128} ReLU for PPO from SLM Lab
# PPO_s 64_64_relu [1.26, 1.16, 2.6 , 1.16, 1.32, 8.16, 6.98, 4.76, 7.48, 8.24, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
# PPO_s 256_128_relu [8.14, 8.14, 8.14, 8.14, 8.14, 5.58, 7.46, 6.5 , 5.7 , 6.8 , 6.8 , 5.5 , 4.28, 4.42, 8.16, 1.26, 2.66, 3.5 , 2.8 , 1.44]
# ===> RanksumsResult(statistic=-3.327160951922825, pvalue=0.0008773568228766609)


# 18. PBRS 3: {256, 128} ReLU for DDPG and PPO from SLM Lab
# DDPG 256_128_relu [8.88, 8.88, 8.88, 8.88, 8.88, 8.36, 8.36, 8.36, 8.36, 8.36, 9.2 , 9.2 , 9.2 , 9.2 , 9.2 , 8.64, 8.64, 8.64, 8.64, 8.64]
# PPO_s 256_128_relu [8.14, 8.14, 8.14, 8.14, 8.14, 5.58, 7.46, 6.5 , 5.7 , 6.8 , 6.8 , 5.5 , 4.28, 4.42, 8.16, 1.26, 2.66, 3.5 , 2.8 , 1.44]
# ===> RanksumsResult(statistic=5.410017808004594, pvalue=6.301848221392269e-08)


# QUBE2

# 1. DDPG vs A2C:
# DDPG 64_64_relu [2.564, 4.112, 2.316, 5.36 , 3.992, 6.26 , 6.276, 6.268, 6.264, 6.248, 6.988, 5.808, 1.892, 6.988, 7.008, 0.532, 0.872, 0.376, 0.156, 0.696]
# A2C 64_64_relu [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
# ===> RanksumsResult(statistic=5.410017808004594, pvalue=6.301848221392269e-08)

# 2. DDPG vs PPO:
# DDPG 64_64_relu [2.564, 4.112, 2.316, 5.36 , 3.992, 6.26 , 6.276, 6.268, 6.264, 6.248, 6.988, 5.808, 1.892, 6.988, 7.008, 0.532, 0.872, 0.376, 0.156, 0.696]
# PPO 64_64_relu [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.452, 0.036, 0.   , 0.24 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]
# ==> RanksumsResult(statistic=5.3288675408845245, pvalue=9.882703709104184e-08)

# 3. QUBE 2 SEED 0 SIM vs REAL:
# SIM [1.9360000000000015, 0.7760000000000006, 3.856000000000003, 3.760000000000003, 4.451999999999953, 5.8079999999998035, 0.10800000000000007, 3.972000000000003, 5.8079999999998035, 5.8439999999998, 2.988000000000002, 2.2080000000000015, 1.0440000000000007, 0.20800000000000016, 3.5640000000000027, 2.672000000000002, 1.5400000000000011, 5.8439999999998, 5.851999999999799, 2.3320000000000016]
# REAL [3.792000000000003, 4.159999999999985, 5.38399999999985, 2.836000000000002, 3.856000000000003, 3.796000000000003, 4.243999999999976, 5.123999999999879, 5.139999999999877, 4.167999999999984, 2.876000000000002, 5.455999999999842, 5.38399999999985, 0.19600000000000015, 3.636000000000003, 4.447999999999953, 5.147999999999876, 1.5120000000000011, 2.3280000000000016, 5.179999999999873]
# ===> RanksumsResult(statistic=1.0414284280408843, pvalue=0.29767675447218034)

# 4. QUBE 2 ALL SEEDS SIM vs REAL:
# SIM [2.564, 4.112, 2.316, 5.36 , 3.992, 6.26 , 6.276, 6.268, 6.264, 6.248, 6.988, 5.808, 1.892, 6.988, 7.008, 0.532, 0.872, 0.376, 0.156, 0.696]
# REAL [2.376, 3.272, 2.048, 5.384, 4.26, 0.34, 0.188, 0.204, 0.496, 0.572, 0.096, 1.148, 0.148, 0.268, 0.12, 0.192, 0.188, 0.216, 0.212, 0.18]
# ===> RanksumsResult(statistic=3.7599623765631924, pvalue=0.0001699389127330722)