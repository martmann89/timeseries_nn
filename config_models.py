import numpy as np
import config as cfg

################### NN - Models ###########################
# Example for configuration
# model_var = {
#     'name': 'Name for identification and plotting',
#     'city': 'city_of_dataset',
#     'type': 'network_type: lstm, convolutional, conv_lstm, lstm_conv, naive',
#     'loss': 'loss type: pinball, quality_driven,
#     'fields': 'list_of_features',
#     'train_bool': 'train_network_bool',
#     'number': 'String number_of_model if same city and type but different weights',
#     'plotting': {'color': 'b/g/r/c/m/y/k/w', 'marker': 'o/v/^/</>/s/*/x/d', 'linestyle': '-/--/-./:'},
#     'baseline': {'type'}
# }

model_pb_mlp = dict(
    name='pinball loss mlp',
    loss='pinball',
    nn_type='mlp',
    train_bool=False,
    epochs=100,
    # alpha=cfg.prediction['alpha'],
    alpha=0.055,
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
    # name='pinball loss LSTMconv',
    name='LSTMconv_PB',
    loss='pinball',
    nn_type='LSTMconv',
    train_bool=False,
    epochs=100,
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
    epochs=1000,
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
#     'city': 'city_of_dataset',
#     'type': 'network_type: lstm, convolutional, conv_lstm, lstm_conv, naive',
#     'loss': 'loss type: pinball, quality_driven,
#     'fields': 'list_of_features',
#     'train_bool': 'train_network_bool',
#     'number': 'String number_of_model if same city and type but different weights',
#     'plotting': {'color': 'b/g/r/c/m/y/k/w', 'marker': 'o/v/^/</>/s/*/x/d', 'linestyle': '-/--/-./:'},
#     'baseline': {'type'}
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
    plotting=dict(
        color='g'
    ),
    bounds=((0.01, None), (0, 1), (0, 1),
            (2.01, 30), (-1, 1)),
    starting_values=np.array([0, 0.5, 0.5,
                              5, -0.5]),
)

model_garch_tv_q = dict(
    name='ARCD_quad',
    plotting=dict(
        color='darkgreen'
    ),
    ### quadratic
    bounds=((0.01, 1), (0, 1), (0, 1),
            (-10, 5), (-5, 5), (-5, 100),
            (-1, 1), (-1, 1), (-1, 1)),
    starting_values=np.array([2, 0.11, 0.9,
                              1, 1, 1,
                              1, 1, 1]),
)

model_garch_tv_c = dict(
    name='ARCD_cos',
    plotting=dict(
        color='limegreen'
    ),
    ### cosine
    bounds=((0.01, None), (0, 1), (0, 1),
            (-10, 10), (-30, 15), (-365, 365),
            (-5, 5), (-8, 8), (-365, 365)),
    starting_values=np.array([0, 0.5, 0.5,
                             -1, 0.5, 35,
                              0, 0.5, 35]),
)

### Naive model
model_naive = dict(
    name='Naive',
    loss='quality_driven',
    # for plotting
    plotting=dict(
        color='r',
        marker='v'
    )
)