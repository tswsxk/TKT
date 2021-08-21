# coding: utf-8
# 2021/5/26 @ tongshiwei
from fire import Fire
from baize import path_append
from EduData import get_data
from TKT import DKT

DATASET = {
    "a0910c": (
        "ktbd-a0910c",
        "a0910c/train.json",
        "a0910c/valid.json",
        "a0910c/test.json",
        {"ku_num": 146, "hidden_num": 100}
    )
}


def get_dataset_and_config(dataset, train=None, valid=None, test=None, hyper_params=None):
    if dataset in DATASET:
        dataset, train, valid, test, hyper_params = DATASET[dataset]

        data_dir = "../../data"
    else:
        data_dir = dataset
        hyper_params = {} if hyper_params is None else hyper_params

    get_data(dataset, data_dir)

    train = path_append(data_dir, train)
    valid = path_append(data_dir, valid)
    test = path_append(data_dir, test)
    return train, valid, test, hyper_params


def dkt(dataset, enable_hyper_search=False):
    train, valid, _, hyper_params = get_dataset_and_config(dataset)
    DKT.benchmark_train(
        train,
        valid,
        hyper_params=hyper_params,
        end_epoch=10,
        enable_hyper_search=enable_hyper_search,
        save=True,
        model_name="dkt",
        model_dir="dkt"
    )


def embed_dkt(dataset, enable_hyper_search=False):
    train, valid, _, hyper_params = get_dataset_and_config(dataset)
    hyper_params.update(dict(add_embedding_layer=True, embedding_dim=35))
    DKT.benchmark_train(
        train,
        valid,
        hyper_params=hyper_params,
        end_epoch=10,
        enable_hyper_search=enable_hyper_search,
        save=True,
        model_name="edkt",
        model_dir="edkt"
    )


def dkt_plus(dataset, enable_hyper_search=False):
    train, valid, _, hyper_params = get_dataset_and_config(dataset)
    DKT.benchmark_train(
        train,
        valid,
        hyper_params=hyper_params,
        end_epoch=10,
        loss_params={"lr": 0.1, "lw1": 0.5, "lw2": 0.5},
        enable_hyper_search=enable_hyper_search,
        save=True,
        model_name="dkt+",
        model_dir="dkt+"
    )


def embed_dkt_plus(dataset, enable_hyper_search=False):
    train, valid, _, hyper_params = get_dataset_and_config(dataset)
    hyper_params.update(dict(add_embedding_layer=True, embedding_dim=35))

    DKT.benchmark_train(
        train,
        valid,
        hyper_params=hyper_params,
        end_epoch=10,
        loss_params={"lr": 0.1, "lw1": 0.5, "lw2": 0.5},
        enable_hyper_search=enable_hyper_search,
        save=True,
        model_name="edkt+",
        model_dir="edkt+"
    )


def train_model(model, dataset, enable_hyper_search=False, *args, **kwargs):
    train, valid, _, hyper_params = get_dataset_and_config(dataset)
    hyper_params.update(dict(add_embedding_layer=True, embedding_dim=35))

    DKT.benchmark_train(
        train,
        valid,
        hyper_params=hyper_params,
        end_epoch=10,
        loss_params={"lr": 0.1, "lw1": 0.5, "lw2": 0.5},
        enable_hyper_search=enable_hyper_search,
        save=True,
        model_name=model,
        model_dir=model
    )


def run(mode, model, dataset, epoch, train_path=None, valid_path=None, test_path=None, embedding_dim=None, *args,
        **kwargs):
    train, valid, test, hyper_params = get_dataset_and_config(dataset, train_path, valid_path, test_path)
    loss_params = {}
    if mode in {"hs", "train"}:
        if model in {"dkt+", "edkt+"}:
            loss_params = {"lr": 0.1, "lw1": 0.5, "lw2": 0.5}
        elif model in {"edkt", "edkt+"}:
            hyper_params.update(dict(add_embedding_layer=True, embedding_dim=embedding_dim))

        DKT.benchmark_train(
            train,
            valid,
            enable_hyper_search=True if mode == "hs" else False,
            end_epoch=epoch,
            loss_params=loss_params,
            hyper_params=hyper_params,
            save=True,
            model_dir=model,
            model_name=model,
            *args, **kwargs
        )
    elif mode == "test":
        print(DKT.benchmark_eval(test, model, epoch))
    else:
        raise ValueError("unknown mode %s" % mode)


if __name__ == '__main__':
    import torch

    torch.manual_seed(0)

    # Fire(run)
    #
    run("train", "dkt", "a0910c", 10)