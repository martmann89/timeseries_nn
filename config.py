"""
This config file should hold all static parameters - everything is changed here (except from the networks structure)
"""

################### PARAMETER for Preprocessing ###########################
data = dict(
        batch_size=60,  # for QD bs > 50
        test_data_size=371,  # 24*365=8760,
        train_data_perc=0.8,
)

# feature to be predicted (in input data)
label = 'd_glo'

# general prediction configs
prediction = dict(
    label=label,
    alpha=0.05,
    horizon=1,
)

d_pred = dict(
    input_len=100,  # data used for parameter calibration, has to be validated
)

nn_pred = dict(
    input_len=6,  # observations taken into account for prediction
    # num_features=1,  # number of input features
)

training = dict(
    max_epochs=100,
    patience=3,
    learning_rate=0.001,  # standard: 0.001
)

plot = dict(
    previous_vals=6,
)