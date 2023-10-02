import numpy as np
import xgboost as xgb
import SFXGBoost
from SFXGBoost.common.XGBoostcommon import PARTY_ID
from SFXGBoost.config import Config, MyLogger, rank

def experiment2():
    seed = 10

    # datasets = ["healthcare", "MNIST", "synthetic", "Census", "DNA", "Purchase-10", "Purhase-20", "Purchase-50", "Purchase-100"]
    # datasets = ["synthetic"]F
    datasets = ["healthcare"]

    # targetArchitectures = ["XGBoost", "FederBoost-central", "FederBoost-federated"]
    targetArchitectures = ["FederBoost-federated"]

    all_data = {} # all_data["XGBoost"]["healthcore"]["accuarcy train shadow"]

    for targetArchitecture in targetArchitectures:
        all_data[targetArchitecture] = {}
        for dataset in datasets:
            config = Config(experimentName = "experiment 2",
            nameTest= dataset + " test",
            model="normal",
            dataset=dataset,
            lam=0.1, # 0.1 10
            gamma=0.5,
            alpha=0.0,
            learning_rate=0.3,
            max_depth=8,
            max_tree=20,
            nBuckets=35,
            save=True,
            target_rank=1)
            logger = MyLogger(config).logger
            if rank == PARTY_ID.SERVER: logger.warning(config.prettyprint())

            np.random.RandomState(seed) # TODO set seed sklearn.split
            
            if targetArchitecture == "FederBoost-central":
                target_model = SFXGBoost(config, logger)
                shadow_model = SFXGBoost(config, logger)
                # attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                attack_model = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                                learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.0, reg_lambda=0.1)
                data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger, targetArchitecture)
                if rank == PARTY_ID.SERVER:
                    all_data[targetArchitecture][dataset] = data
            elif targetArchitecture == "FederBoost-federated":
                target_model = SFXGBoost(config, logger)
                shadow_models = [SFXGBoost(config, logger) for _ in range(10)]  # TODO create the amount of shadow models allowed given by datasetretrieval
                attack_model= None  # TODO define federated neural network.
                data = train_all_federated(target_model, shadow_models, attack_model, config, logger)
                if rank == PARTY_ID.SERVER:
                    all_data[targetArchitecture][dataset] = data
            elif targetArchitecture == "XGBoost" and rank == PARTY_ID.SERVER:  # define neural network
                target_model = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx", num_class=config.nClasses,
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                shadow_model = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx", num_class=config.nClasses,
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                # attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                attack_model = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                            learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.0, reg_lambda=0.1)
                data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger, targetArchitecture)
                all_data[targetArchitecture][dataset] = data
            else:
                continue
                # raise Exception("Wrong model types given!")
            
            # saver.save_experiment_2_one_run(attack_model, )
    if rank == PARTY_ID.SERVER:
        save(all_data, name="all_data federated", config=config)
    plot_experiment2(all_data)
        
        # saver.create_plots_experiment_2(all_data)


def get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger, name="sample"):
    # TODO keep track of rank 
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    log_distribution(logger, X_train, y_train, y_test)
    D_Train_Shadow, D_Out_Shadow = split_shadow((X_shadow, y_shadow)) 
    if type(target_model) == SFXGBoost:
        target_model.fit(X_train, y_train, fName) # chrashes
        shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)
    else:
        target_model.fit(X_train, np.argmax(y_train, axis=1))
        shadow_model.fit(D_Train_Shadow[0], np.argmax(D_Train_Shadow[1], axis=1))
    z, labels = create_D_attack_centralised(shadow_model, D_Train_Shadow, D_Out_Shadow)
    data = None
    if rank == PARTY_ID.SERVER:
        attack_model.fit(z, labels)
        z_test, labels_test = create_D_attack_centralised(target_model, (X_train, y_train), (X_test, y_test))
        
        data = retrieve_data(target_model, shadow_model, attack_model, X_train, y_train, X_test, y_test, z_test, labels_test)
        # plot_auc(labels_test, attack_model.predict_proba(z_test)[:, 1], f"Plots/experiment2_centralised_AUC_{name}.jpeg")
    return data