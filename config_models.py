import numpy as np
import config as cfg

################### NN - Models ###########################
# Example for configuration
# model_var = {
#     'name': Name for identification and plotting,
#     'loss': Name loss function 'pinball' or 'quality_driven'
#     'nn_type': network_type 'mlp' or 'LSTMconv'
#     'train_bool': train network 'True' or 'False',
#     'epochs': number of epochs network is traines with,
#     alpha: confidence level,
#     conf_int: conformal prediction int approach 'True' or 'False',
#     conf_alpha = conformal prediction alpha approach 'None' or 'list of alphas',
#     'plotting': {'color': 'b/g/r/c/m/y/k/w', 'marker': 'o/v/^/</>/s/*/x/d', 'linestyle': '-/--/-./:'},
# }

model_pb_mlp = dict(
    name='pinball loss mlp',
    loss='pinball',
    nn_type='mlp',
    train_bool=False,
    epochs=3, #100
    alpha=cfg.prediction['alpha'],
    # alpha=0.055,
    conf_int=False,
    conf_alpha=None,
    # conf_alpha=np.linspace(cfg.prediction['alpha'], cfg.prediction['alpha']/10, 10),
    ### for plotting
    plotting=dict(
        color='orangered',
        marker='*',
    )
)

model_pb_lstm = dict(
    name='LSTMconv_PB',
    loss='pinball',
    nn_type='LSTMconv',
    train_bool=False,
    epochs=3, #100
    alpha=cfg.prediction['alpha'],
    conf_int=False,
    conf_alpha=None,
    # conf_alpha=np.linspace(cfg.prediction['alpha'], cfg.prediction['alpha']/10, 10),
    ### for plotting
    plotting=dict(
        color='gold',
        marker='*',
    )
)

model_qd = dict(
    name='MLP_QD',
    loss='quality_driven',
    nn_type='mlp',
    train_bool=False,
    epochs=3,#1000
    alpha=cfg.prediction['alpha'],
    conf_int=False,
    conf_alpha=None,
    # for plotting
    plotting=dict(
        color='purple',
        marker='*',
    )
)

################### Distribution - Models ###########################
# Example for configuration
# model_var = {
#     'name': 'Name for identification and plotting',
#     'bounds': Boundaries for parameter of garch fit
#     'starting_values': starting values of parameter of garch fit
#     'plotting': {'color': 'b/g/r/c/m/y/k/w', 'marker': 'o/v/^/</>/s/*/x/d', 'linestyle': '-/--/-./:'},
# }

model_garch = dict(
    name='GARCH',
    # for plotting
    plotting=dict(
        color='cyan'
    )
)

model_my_garch = dict(
    name='my_GARCH',
    bounds=((0.01, None), (0, 1), (0, 1),
            (2.01, 30), (-1, 1)),
    starting_values=np.array([0, 0.5, 0.5,
                              5, -0.5]),
    plotting=dict(
        color='g'
    ),
)

model_garch_tv_q = dict(
    name='ARCD_quad',
    ### quadratic
    bounds=((0.01, 1), (0, 1), (0, 1),
            (-10, 5), (-5, 5), (-5, 100),
            (-1, 1), (-1, 1), (-1, 1)),
    starting_values=np.array([2, 0.11, 0.9,
                              1, 1, 1,
                              1, 1, 1]),
    plotting=dict(
        color='darkgreen'
    ),
)

model_garch_tv_c = dict(
    name='ARCD_cos',
    ### cosine
    bounds=((0.01, None), (0, 1), (0, 1),
            (-10, 10), (-30, 15), (-365, 365),
            (-5, 5), (-8, 8), (-365, 365)),
    starting_values=np.array([0, 0.5, 0.5,
                             -1, 0.5, 35,
                              0, 0.5, 35]),
    plotting=dict(
        color='limegreen'
    ),
)

### Naive model
model_naive = dict(
    name='Naive',
    # for plotting
    plotting=dict(
        color='r',
        marker='v'
    )
)