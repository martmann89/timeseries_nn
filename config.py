"""
This config file should hold all static parameters - everything is changed here (except from the networks structure)
"""

################### PARAMETER for Preprocessing ###########################
data = dict(
        batch_size=60,  # for QD bs > 50
        test_data_size=365,
        # test_data_size=10,
        train_data_perc=0.8,
        # type='simulated_data'
        type='pv_data'
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
    lom='cos',  # Law of Motion (cos or quad)

    length=1500,

    ####### GARCH(1,1)
    alpha0=0.3,
    alpha1=0.2,
    beta1=0.7,
    ### estimated parameter
    # alpha0=0.01367,
    # alpha1=0.10238,
    # beta1=0.89762,

    ######### time-varying dof (eta)
    ### Quadratic parameter
    # eta1=-2,
    # eta2=1,
    # eta3=-0.1,
    ### Cosine parameter
    eta1=-2,
    eta2=1,
    eta3=42,
    ### estimated parameter
    # eta1=-5.25822,
    # eta2=-2.81094,
    # eta3=60.86728,

    # time-varying skewness (lambda)
    ### Quadratic parameter
    # lam1=-0.45,
    # lam2=0.4,
    # lam3=0.1,
    ### Cosine parameter
    lam1=-0.35,
    lam2=0.7,
    lam3=42,
    ### estimated parameter
    # lam1=-0.03241,
    # lam2=0.12165,
    # lam3=-0.05193,
)

monte_carlo = 2000

seed = 987654321  # 123456789
