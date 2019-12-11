# coding: utf-8
# 2019/12/11 @ tongshiwei


from TKT.DKT import DKT


def test_train(root_data_dir, dataset):
    # test for DKT
    DKT.run(
        [
            "train", "$data_dir/train.json", "$data_dir/test.json",
            "--workspace", "DKT",
            "--hyper_params",
            "nettype=DKT;ku_num=int(50);hidden_num=int(100);dropout=float(0.5)",
            "--end_epoch", "int(1)",
            "--ctx", "cpu",
            "--dataset", dataset,
            "--root_data_dir", root_data_dir,
            "--data_dir", "$root_data_dir"
        ]
    )

    # test for DKT+
    try:
        DKT.run(
            [
                "train", "$data_dir/train.json", "$data_dir/test.json",
                "--workspace", "DKT+",
                "--hyper_params",
                "nettype=DKT;ku_num=int(50);hidden_num=int(100);latent_dim=int(35);dropout=float(0.5)",
                "--loss_params", "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)",
                "--end_epoch", "int(1)",
                "--ctx", "cpu",
                "--dataset", dataset,
                "--root_data_dir", root_data_dir,
                "--data_dir", "$root_data_dir"
            ]
        )
    except ValueError:
        assert True

    # test for EmbedDKT
    DKT.run(
        [
            "train", "$data_dir/train.json", "$data_dir/test.json",
            "--workspace", "DKT",
            "--hyper_params",
            "nettype=EmbedDKT;ku_num=int(50);hidden_num=int(100);latent_dim=int(35);dropout=float(0.5)",
            "--end_epoch", "int(1)",
            "--ctx", "cpu",
            "--dataset", dataset,
            "--root_data_dir", root_data_dir,
            "--data_dir", "$root_data_dir"
        ]
    )

    # test for EmbedDKT+
    try:
        DKT.run(
            [
                "train", "$data_dir/train.json", "$data_dir/test.json",
                "--workspace", "EmbedDKT+",
                "--hyper_params",
                "nettype=EmbedDKT;ku_num=int(50);hidden_num=int(100);latent_dim=int(35);dropout=float(0.5)",
                "--loss_params", "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)",
                "--end_epoch", "int(1)",
                "--ctx", "cpu",
                "--dataset", dataset,
                "--root_data_dir", root_data_dir,
                "--data_dir", "$root_data_dir"
            ]
        )
    except ValueError:
        assert True
