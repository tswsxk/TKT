```bash
python3 run.py train ~/TKT/data/\$dataset/data/train ~/TKT/data/\$dataset/data/test --root ~/TKT --workspace DKT  --hyper_params "nettype=DKT;ku_num=int(146);hidden_num=int(200);dropout=float(0.5)" --dataset assistment0910c --batch_size "int(16)" --ctx "cuda:0" --optimizer_params "lr=float(1e-2)"
```

```bash
python3 DKT.py train ~/TKT/data/\$dataset/data/train ~/TKT/data/\$dataset/data/test --root ~/TKT --workspace DKT  --hyper_params "nettype=DKT;ku_num=int(146);hidden_num=int(200);dropout=float(0.5)" --dataset assistment0910c --batch_size "int(16)" --ctx "cuda:0" --optimizer_params "lr=float(1e-2)"
```


```bash
python3 DKT.py train \$data_dir/train \$data_dir/test --workspace EmbedDKT  --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --dataset="junyi" --ctx="cuda:0"
```

```bash
python3 DKT.py train \$data_dir/train_0 \$data_dir/valid_0 --workspace DKT+  --hyper_params "nettype=DKT+;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --dataset="junyi_100" --ctx="cuda:0" --loss_params "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)"
```

```bash
export PYTHONPATH=$PYTHONPATH:~/TKT
python3 DKT.py train \$data_dir/train_\$caption \$data_dir/valid_\$caption --root ~/TKT  --hyper_params "nettype=DKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --ctx cuda:5 --caption 0 --workspace DKT_0 --dataset="junyi_2000"
```