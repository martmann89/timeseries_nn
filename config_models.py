import numpy as np

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
    name='pinball loss',
    loss='pinball',
    train_bool=False,
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
    bounds=((0.0001, None), (0, None), (0, None),
            (-2, 2), (-2, 2), (-2, 2),
            (-1, 1), (-1, 1), (-1, 1)),
    starting_values=np.array([5.6, 0.11, 0.9,
                              1, 1, 1,
                              1, 1, 1]),
    # bounds=((10, None), (0, None), (0, None),
    #         (2.001, 30), (-0.999, 0.999)),
    # starting_values=np.array([11, 0.11, 0.9,
    #                           3,
    #                           0.1]),
    # dist='skst',

)
