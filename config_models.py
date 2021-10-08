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
    name='pinball_loss',
    loss='pinball',
    train_bool=False,
    epochs=100,
    alpha=cfg.prediction['alpha'],
    conf_int=False,
    conf_alpha=None,
    # conf_alpha=np.linspace(cfg.prediction['alpha'], cfg.prediction['alpha']/10, 10),
    # for plotting
    plotting=dict(
        color='g'
    )
)
model_qd = dict(
    name='quality-driven_loss',
    loss='quality_driven',
    train_bool=False,
    epochs=1000,
    alpha=cfg.prediction['alpha'],
    conf_int=False,
    conf_alpha=None,
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
    bounds=((0.01, None), (0, 1), (0, 1),
            (-5, 5), (-5, 5), (-2, 2),
            (-1, 1), (-1, 1), (-1, 1)),
    starting_values=np.array([2, 0.11, 0.9,
                              1, 1, 1,
                              1, 1, 1]),
    # bounds=((0.01, None), (0, 1), (0, 1),
    #         (-5, 5), (-15, 15), (0, 365),
    #         (-5, 5), (-8, 8), (0, 365)),
    # starting_values=np.array([0, 0.5, 0.5,
    #                          -1, 0.5, 35,
    #                           0, 0.5, 35]),

)
