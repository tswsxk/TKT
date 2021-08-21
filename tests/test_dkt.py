# coding: utf-8
# 2019/12/11 @ tongshiwei

from TKT import DKT
from TKT.DKT import etl


def test_dkt(tmpdir, root_data_dir, train_dataset, test_dataset):
    model_dir = str(tmpdir.mkdir("dkt"))

    model = DKT(init_net=True, hyper_params={"ku_num": 50, "hidden_num": 100}, model_dir=model_dir, model_name="dkt")

    model.train(
        etl(train_dataset, model.cfg),
        etl(test_dataset, model.cfg),
        save=False,
        end_epoch=2,
    )

    model.save(model.cfg.model_dir)

    model = DKT.from_pretrained(model_dir)

    model.eval(etl(test_dataset, model.cfg))


def test_benchmark(tmpdir, root_data_dir, train_dataset, test_dataset):
    model_dir = str(tmpdir.mkdir("dkt"))

    DKT.benchmark_train(
        train_dataset,
        test_dataset,
        enable_hyper_search=True,
        save=False,
        model_dir=model_dir,
        model_name="dkt",
        end_epoch=1,
        hyper_params={"ku_num": 50, "hidden_num": 100}
    )

    DKT.benchmark_train(
        train_dataset,
        test_dataset,
        save=True,
        model_dir=model_dir,
        model_name="dkt",
        end_epoch=2,
        hyper_params={"ku_num": 50, "hidden_num": 100}
    )
    model = DKT.from_pretrained(model_dir, 1)
    model.benchmark_eval(test_dataset, model_dir, 1)
