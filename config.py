"""
This config file should hold all static parameters - everything is changed here (except from the networks structure)
"""

################### PARAMETER for Preprocessing ###########################
data = dict(
        batch_size=60,  # for QD bs > 50
        test_data_size=365,  # 24*365=8760,
        train_data_perc=0.807,
        type='simulated_data'
        # type='pv_data'
)

# feature to be predicted (in input data)
label = 'd_glo'

# general prediction configs
prediction = dict(
    label=label,
    alpha=0.1,
    horizon=1,
)

d_pred = dict(
    input_len=1000,  # data used for parameter calibration, has to be validated
    # input_len=1500,  # data used for parameter calibration, has to be validated
)

nn_pred = dict(
    input_len=6,  # observations taken into account for prediction
    # num_features=1,  # number of input features
)

training = dict(
    # max_epochs=1000,
    patience=3,
    learning_rate=0.001,  # standard: 0.001
)

plot = dict(
    previous_vals=6,
)

################# parameter for data generation #############
data_gen = dict(
    length=1500,

    # GARCH(1,1)
    alpha0=0.04,
    alpha1=0.2,
    beta1=0.8,

    # time-varying dof (eta)
    eta1=-2,
    eta2=-0.54,
    eta3=-0.08,

    # time-varying skewness (lambda)
    lam1=-0.05,
    lam2=0.2,
    lam3=0.1,
)

monte_carlo = 1000
