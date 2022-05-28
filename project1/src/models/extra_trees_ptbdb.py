from sklearn.ensemble import ExtraTreesClassifier

PARAMS = {"n_estimators": 855, "criterion": "entropy", "max_depth": 50}


def get_model(params=PARAMS):
    return ExtraTreesClassifier(n_estimators=params["n_estimators"],
                                criterion=params["criterion"],
                                max_depth=params["max_depth"],
                                n_jobs=-1,
                                verbose=1)
