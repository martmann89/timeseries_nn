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

model_pb = dict(
    name='pinball_loss_ens',
    loss='pinball',
    train_bool=True,
    alpha=cfg.prediction['alpha'],
    conf_int=False,
    conf_alpha=np.linspace(cfg.prediction['alpha'], cfg.prediction['alpha']/10, 10),
    # for plotting
    plotting=dict(
        color='g'
    )
)
model_qd = dict(
    name='quality-driven loss',
    loss='quality_driven',
    train_bool=False,
    # for plotting
    plotting=dict(
        color='r'
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
        color='y'
    )
)

model_garch_tv = dict(
    name='GARCH_tv',
    plotting=dict(
        color='g'
    ),
    # bounds=((0.01, None), (0, None), (0, None),
    #         (-2, 2), (-2, 2), (-2, 2),
    #         (-1, 1), (-1, 1), (-1, 1)),
    # starting_values=np.array([5.6, 0.11, 0.9,
    #                           1, 1, 1,
    #                           1, 1, 1]),
    bounds=((0.01, None), (0, 1), (0, 1),
            (-4, 0), (-1, 1), (-0.25, 0.25),
            (-0.5, 0.5), (-0.5, 0.5), (-0.25, 0.25)),
    starting_values=np.array([2, 0.5, 0.5,
                             -1, -0.5, 0,
                              0, 0.1, 0]),

)
