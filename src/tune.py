"""Tune the model parameters."""
import json
from pathlib import Path

import yaml
from ray import tune
from ray.tune.integration.xgboost import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils import get_model_path, load_config, load_data


def get_tune_param_space(tune_config: dict):
    """Return the parameter space to tune.

    Parameters
    ----------
    tune_config : dict
        The configuration dict for the tune stage.
    """
    hyperparams = config["hyperparams"]
    return {
        "objective": hyperparams["objective"],
        "tree_method": hyperparams["tree_method"],
        "early_stopping_rounds": hyperparams["early_stopping_rounds"],
        "eval_metric": hyperparams["eval_metric"],
        "n_estimators": tune.randint(200, 600),
        "gamma": tune.randint(1, 5),
        "max_depth": tune.randint(2, 9),
        "min_child_weight": tune.randint(1, 5),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
        "colsample_bytree": tune.uniform(0.5, 1),
    }


def get_objective(X_train, y_train):
    """Return the objective function to be optimized."""

    def objective(config):
        """Objective to be optimized.

        Uses a simple 0.8/0.2 train-validation-split and logs the
        validation logloss using the `TuneReportCallback`.

        Parameters
        ----------
        config: dict
            The config object.
        """
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, stratify=y_train, shuffle=True, test_size=0.2
        )
        trc = TuneReportCallback({"loss": "validation_0-mlogloss"})
        XGBClassifier(**config, callbacks=[trc]).fit(
            X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], verbose=False
        )

    return objective


def run_tune(config: dict):
    """Train the model."""
    wines = load_data(raw=False, config=yaml.safe_load(open("params.yaml"))["prepare"])
    X, y = wines.drop("quality", axis=1), wines["quality"]
    X = X.astype("float32")
    y = y.astype("long")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # apply standard scaling
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    objective = get_objective(X_train, y_train)
    param_space = get_tune_param_space(config)
    search_alg = OptunaSearch()
    scheduler = ASHAScheduler(
        grace_period=config["scheduler"]["grace_period"],
        reduction_factor=config["scheduler"]["reduction_factor"],
    )
    tuner = tune.Tuner(
        objective,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=config["tune_config"]["num_samples"],
            metric=config["tune_config"]["metric"],
            mode=config["tune_config"]["mode"],
            scheduler=scheduler,
            search_alg=search_alg,
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result("loss", mode="min")
    best_params = best_result.config

    best_result.metrics_dataframe[["training_iteration", "loss"]].to_csv(
        "eval/losses.csv", index=False
    )

    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    json.dump(best_params, open("eval/params.json", "w"), indent=4)

    clf = XGBClassifier(**best_params).fit(
        X_train, y_train, eval_set=[(X_test, y_test)], verbose=50
    )
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    clf.save_model(get_model_path(config))

    acc = clf.score(X_test, y_test)
    json.dump({"accuracy": acc}, open("eval/metrics.json", "w"), indent=4)


if __name__ == "__main__":
    config = load_config("tune")
    run_tune(config)
