"""
This config file should hold all static parameters - everything is changed here (except from the networks structure)
"""

################### PARAMETER for Preprocessing ###########################
data = dict(
        batch_size=60,  # for QD bs > 50
        test_data_size=371,  # 24*365=8760,
        train_data_perc=0.8,
)

label = 'd_glo'

prediction = dict(
    pos=0,
    num_predictions=1,
    input_len=6,
    num_features=1,
    label=label,
)

intervals = dict(
    alpha=0.05
)

training = dict(
    max_epochs=50,
    patience=3,
    learning_rate=0.001,  # standard: 0.001
)

garch = dict(
    input_len=100,
)
