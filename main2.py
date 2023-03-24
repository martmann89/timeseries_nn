import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator import mod

from NN_models.neural_networks import run_nn
from naive_model import naive_prediction

import utility
from utility import print_mean_stats, plot_intervals, plot_intervals2
import config_models as cfg_mod
import config as cfg
from utility import ROOT_DIR
import os

def main():
    ###################################### data import #################################
    ### Real World Data
    
    #print(data.head(n=40))
    #data /= np.std(data)
    #print(data.head(n=40))
    # df = pd.DataFrame(data, columns=['d_glo'])
    # df['#day'] = list(map(lambda x: mod(x, 365), range(len(df))))
    # plt.figure(figsize=(12, 3))
    # plt.plot(range(0,df.size-60),df[0:df.size-60])
    # plt.plot(range(df.size-60,df.size), df[df.size-60:df.size])
    # plt.show()
    
    doTrain = True
    epochs = [10]
    prevs = [6]
    count = 1
    
    dataSources = ["TTF_FM_old","TTF_FM_new"]
    testDataSize = [165, 60]
    alphas = [0.05, 0.95]

    for k in range(0,2):
        cfg.prediction["alpha"] = alphas[k]
        for j in range(0,2):
            data = pd.read_csv(ROOT_DIR + f'/data/{dataSources[j]}.csv',sep=";")
            df = pd.DataFrame(data, columns=["Price"])

            df_res = pd.DataFrame(columns=["Run","PICP_LSTM","MPIW_LSTM","PICP_MLP","MPIW_MLP"])
            df_res.astype({'Run': 'int32'})

            cfg.data["test_data_size"] = testDataSize[j]

            for i in range(1,count+1):
                for nof_epochs in epochs:
                    cfg_mod.model_pb_lstm["epochs"] = nof_epochs
                    cfg_mod.model_pb_mlp["epochs"] = nof_epochs
                    for nof_prevs in prevs:
                        cfg.nn_pred["input_len"] = nof_prevs
                        cfg.plot["previous_vals"] = nof_prevs
                    ### Neural network models
                        model_pb_mlp = cfg_mod.model_pb_mlp
                        model_pb_lstm = cfg_mod.model_pb_lstm
                        model_qd = cfg_mod.model_qd
                        if doTrain:
                            model_pb_mlp["train_bool"] = True
                            model_pb_lstm["train_bool"] = True
                            model_qd["train_bool"] = True
                        else:
                            model_pb_mlp["train_bool"] = False
                            model_pb_lstm["train_bool"] = False
                            model_qd["train_bool"] = False

                        ### naive model
                        model_naive = cfg_mod.model_naive

                        
                        ### Run NN Models
                        model_pb_mlp = run_nn(df, model_pb_mlp)
                        model_pb_lstm = run_nn(df, model_pb_lstm)
                        #model_qd = run_nn(df, model_qd)

                        ### Run naive model
                        model_naive = naive_prediction(model_naive, df, 17)

                        ################## Plot prediction intervals (multiple forecasts) ################
                        ### Plot 1 = 17, Plot 2 = 175, Plot 3 = 210
                        start_idx = 0  # between 0 and 365
                        end_idx = cfg.data["test_data_size"]
                        plt.figure(figsize=(24, 24))
                        #plot_intervals2(model_qd, start_idx, end_idx)
                        plot_intervals2(model_pb_mlp, start_idx, end_idx)
                        plot_intervals2(model_pb_lstm, start_idx, end_idx)
                        # plot_intervals2(model_naive, start_idx, end_idx)
                        os.makedirs(f'./predictions/{testDataSize[j]}d_{dataSources[j]}/mlp', exist_ok=True)
                        res = pd.DataFrame(model_pb_mlp['intervals'],columns=["lower","upper","mean"])
                        res.to_csv(f'./predictions/{testDataSize[j]}d_{dataSources[j]}/mlp/intervals_epochs_{nof_epochs}_prevs_{nof_prevs}_alpha_{str(cfg.prediction["alpha"]).replace(".","_")}_run_{i}.csv',sep=";", index=False)
                        os.makedirs(f'./predictions/{testDataSize[j]}d_{dataSources[j]}/lstm', exist_ok=True)
                        res = pd.DataFrame(model_pb_lstm['intervals'], columns=["lower","upper","mean"])
                        res.to_csv(f'./predictions/{testDataSize[j]}d_{dataSources[j]}/lstm/intervals_epochs_{nof_epochs}_prevs_{nof_prevs}_alpha_{str(cfg.prediction["alpha"]).replace(".","_")}_run_{i}.csv',sep=";", index=False)
                        plt.plot(range(cfg.nn_pred["input_len"] + 1, cfg.nn_pred["input_len"] + cfg.data["test_data_size"] + 1), df[(df.size-cfg.data["test_data_size"]):df.size])

                        plt.title(f'Epochs : {nof_epochs}, Nof prev. values accounted: {nof_prevs}, Pinball {1 - cfg.prediction["alpha"]}, Run : {i}')
                        path1 = f'./predictions/{testDataSize[j]}d_{dataSources[j]}/plots'
                        if not os.path.exists(path1):
                            os.makedirs(path1)
                        plt.savefig(path1 + f'/epochs_{nof_epochs}_prevs_{nof_prevs}_alpha_{str(cfg.prediction["alpha"]).replace(".","_")}_run_{i}')
                        # if not os.path.exists(path2):
                        #     os.makedirs(path2)
                        # plt.savefig(path2 + f'/prevs_{nof_prevs}')
                        picp_lstm, mpiw_lstm = print_mean_stats(model_pb_lstm)
                        picp_mlp, mpiw_mlp = print_mean_stats(model_pb_mlp)
                        df_res.loc[len(df_res)] =  [i,picp_lstm,mpiw_lstm,picp_mlp,mpiw_mlp]
            df_res.to_csv(f'./predictions/{testDataSize[j]}d_{dataSources[j]}/results_alpha_{str(cfg.prediction["alpha"]).replace(".","_")}.csv', sep=";", index=False)


if __name__ == '__main__':
    main()