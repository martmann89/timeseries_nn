################### Models ###########################
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

# type_of_loss = 'pinball'
type_of_loss = 'quality_driven'

pb_model = dict(
    name=type_of_loss + ' loss model',
    loss=type_of_loss,
    train_bool=True,
)
