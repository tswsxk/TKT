# TKT

[![PyPI](https://img.shields.io/pypi/v/TKT.svg)](https://pypi.python.org/pypi/TKT)
[![Build Status](https://www.travis-ci.org/tswsxk/TKT.svg?branch=master)](https://www.travis-ci.org/tswsxk/TKT)
[![codecov](https://codecov.io/gh/tswsxk/TKT/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/TKT)

Multiple Knowledge Tracing models implemented by mxnet-gluon. 
For convenient dataset downloading and preprocessing of knowledge tracing task, 
visit [Edudata](https://github.com/bigdata-ustc/EduData) for handy api.

Visit https://base.ustc.edu.cn for more of our works.

## Performance in well-known Dataset

With [`EduData`](https://pypi.python.org/pypi/EduData), we test the models performance, the AUC result is listed as follows:

|model name  | synthetic | assistment_2009_2010 | junyi |
| ---------- | - |------------------ | ----- |
| DKT        | 0.6438748958881487 | 0.7442573465541942 | 0.8305416859735839 |
| DKT+       | **0.8062221383790489** | 0.7483424087919035 | **0.8497422607539136** |
| EmbedDKT   | 0.4858168704660636 | 0.7285572301977586 | 0.8194401881889697 |
| EmbedDKT+ | 0.7340996181876187 | **0.7490900876356051** |0.8405445812109871|
| DKVMN | TBA | TBA |TBA|

The f1 scores are listed as follows:

|model name  | synthetic | assistment_2009_2010 | junyi |
| ---------- | ------------------ | ----- | ----- |
| DKT        | 0.5813237474584396 | 0.7134380508024369 | 0.7732850122818582 |
| DKT+       | **0.7041804463370387** | **0.7137627713343819** | **0.7928075377114897** |
| EmbedDKT   | 0.4716821311199386     | 0.7095025134079656 | 0.7681817174082963 |
| EmbedDKT+   | 0.6316953625658291 | 0.7101790604990228 | 0.7903592922756097 |
| DKVMN | TBA       | TBA                  | TBA   |

The information of the benchmark datasets can be found in EduData docs.

In addition, all models are trained 20 epochs with `batch_size=16`, where the best result is reported.  We use `adam` with `learning_rate=1e-3`. We also apply `bucketing` to accelerate the training speed. Moreover, each sample length is limited to 200. The hyper-parameters are listed as follows:

|model name  | synthetic - 50 | assistment_2009_2010 - 124 | junyi-835 |
| ---------- | ------------------ | ----- | ----- |
| DKT        | `hidden_num=int(100);dropout=float(0.5)` | `hidden_num=int(200);dropout=float(0.5)` | `hidden_num=int(900);dropout=float(0.5)` |
| DKT+       | `lr=float(0.2);lw1=float(0.001);lw2=float(10.0)` | `lr=float(0.1);lw1=float(0.003);lw2=float(3.0)` | `lr=float(0.01);lw1=float(0.001);lw2=float(1.0)` |
| EmbedDKT   | `hidden_num=int(100);latent_dim=int(35);dropout=float(0.5)` | `hidden_num=int(200);latent_dim=int(75);dropout=float(0.5)` | `hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)` |
| EmbedDKT+   | `lr=float(0.2);lw1=float(0.001);lw2=float(10.0)` | `lr=float(0.1);lw1=float(0.003);lw2=float(3.0)` | `lr=float(0.01);lw1=float(0.001);lw2=float(1.0)` |
| DKVMN      | `hidden_num=int(50);key_embedding_dim=int(10);value_embedding_dim=int(10);key_memory_size=int(5);key_memory_state_dim=int(10);value_memory_size=int(5);value_memory_state_dim=int(10);dropout=float(0.5)` | `hidden_num=int(50);key_embedding_dim=int(50);value_embedding_dim=int(200);key_memory_size=int(50);key_memory_state_dim=int(50);value_memory_size=int(50);value_memory_state_dim=int(200);dropout=float(0.5)` | `hidden_num=int(600);key_embedding_dim=int(50);value_embedding_dim=int(200);key_memory_size=int(20);key_memory_state_dim=int(50);value_memory_size=int(20);value_memory_state_dim=int(200);dropout=float(0.5)` |

The number after `-` in the first row indicates the knowledge units number in the dataset. The datasets we used can be  either found in [basedata-ktbd](http://base.ustc.edu.cn/data/ktbd/) or be downloaded by:

```shell
pip install EduData
edudata download ktbd
```

### Trick

* DKT: `hidden_num` is usually set to the nearest hundred number to the `ku_num`
* EmbedDKT: `latent_dim` is usually set to a value litter than or equal to `\sqrt(hidden_num * ku_num)`
* DKVMN: `key_embedding_dim = key_memory_state_dim` and `value_embedding_dim = value_memory_state_dim`

### Notice
Some interfaces of pytorch may change with version changing, such as
```python
import torch
torch.nn.functional.one_hot
```
which may caused some errors like:
```shell
AttributeError: module 'torch.nn.functional' has no attribute 'one_hot'
```

Except that, there is a known bug `Segmentation fault: 11`:
```shell
Segmentation fault: 11

Stack trace:
  [bt] (0) /usr/local/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2e6b160) [0x7f3e4b5b6160]
  [bt] (1) /lib64/libc.so.6(+0x36340) [0x7f3ec3c89340]
  [bt] (2) /usr/local/lib/python3.6/site-packages/torch/lib/libtorch.so(+0x40a5760) [0x7f3dc265c760]
  [bt] (3) /usr/local/lib/python3.6/site-packages/torch/lib/libtorch.so(+0x40a35c5) [0x7f3dc265a5c5]
  [bt] (4) /lib64/libstdc++.so.6(+0x5cb19) [0x7f3eb807db19]
  [bt] (5) /lib64/libc.so.6(+0x39c29) [0x7f3ec3c8cc29]
  [bt] (6) /lib64/libc.so.6(+0x39c77) [0x7f3ec3c8cc77]
  [bt] (7) /lib64/libc.so.6(__libc_start_main+0xfc) [0x7f3ec3c7549c]
  [bt] (8) python3() [0x41da20]
```
However, the mentioned-above bug does not affect the train and evaluation.

PS. if you think those problems are so easy to solve, please do not hesitate to contact us :-).


## Tutorial

### Installation

1. First get the repo in your computer by `git` or any way you like.
2. Suppose you create the project under your own `home` directory, then you can use use 
    1. `pip install -e .` to install the package, or
    2. `export PYTHONPATH=$PYTHONPATH:~/TKT`
    
### Data Format
In `TKT`, all sequence is store in `json` format, such as:
```json
[[419, 1], [419, 1], [419, 1], [665, 0], [665, 0]]
```
Each item in the sequence represent one interaction. The first element of the item is the exercise id 
and the second one indicates whether the learner correctly answer the exercise, 0 for wrongly while 1 for correctly  
One line, one `json` record, which is corresponded to a learner's interaction sequence.

A demo loading program is presented as follows:
```python
import json
from tqdm import tqdm

def extract(data_src):
    responses = []
    step = 200
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)
            for i in range(0, len(data), step):
                if len(data[i: i + step]) < 2:
                    continue
                responses.append(data[i: i + step])

    return responses
```
The above program can be found in `TKT/TKT/shared/etl.py`.

To deal with the issue that the dataset is store in `tl` format:

```text
5
419,419,419,665,665
1,1,1,0,0
```

Refer to [Edudata Documentation](https://github.com/bigdata-ustc/EduData#format-converter).


### CLI

#### General Command Format

---

All command to invoke the model has the same cli canonical form:
```shell
python Model.py $subcommand $parameters1 $parameters2 ...
```
There are several options for the subcommand, use `--help` to see more options and the corresponding parameters:
```shell
python Model.py --help
python Model.py $subcommand --help 
```
The cli tools is constructed based on [longling ConfigurationParser](https://longling.readthedocs.io/zh/latest/submodule/lib/index.html#module-longling.lib.parser).

#### Demo

---

As an example, suppose you create the project under your own `home` directory 
and create a `data` directory to store the data (like `train` and `test`) and model. 
Assume that you are going to test the models on [ktbd](http://base.ustc.edu.cn/data/ktbd/) dataset, 
and the toc of the project is looked like as follows:

```text
└── XKT/                            
    ├── data/
    │   └── ktbd/                
    │        ├── junyi/             <-- dataset
    │        │   ├── train.json
    │        │   └── test.json
    │        ├── ...
    │        └── synthetic/
    ├── ...
    └── XKT/
```

And in each dataset, `train.json` is the training dataset, and `test.json` is the test dataset, 
we want the model is placed under the corresponding dataset directory,
where a `model` directory is created to store the all models. Thus, we use the following command to train the model


```shell
# basic
python3 DKT.py train $HOME/XKT/data/ktbd/junyi/train.json $HOME/XKT/data/ktbd/junyi/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --ctx="gpu(0)" --model_dir $HOME/XKT/data/ktbd/junyi/model/DKT 
# advanced path configuration
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --ctx="gpu(0)" --model_name DKT --root=$HOME/XKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
```
And we can get something like that:
```text
junyi/
├── model/
│   └── DKT/
│       ├── configuration.json
│       ├── DKT-0001.parmas
│       ├── DKT-0002.parmas
│       ├── ...
│       ├── DKT-0020.parmas
│       ├── result.json
│       └── result.log
├── test.json
└── train.json
```
The two command mentioned above are equally the same. 
About how to use the advanced path configuration, 
refer to [longling doc](https://longling.readthedocs.io/zh/latest/submodule/ML/index.html#configuration).

---

### DKT
```shell
# DKT
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --ctx="cuda:0" --model_name DKT --root=$HOME/TKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
# DKT+
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=DKT;ku_num=int(835);hidden_num=int(900);dropout=float(0.5)" --loss_params "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)" --ctx="cuda:0" --model_name DKT+ --root=$HOME/TKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
```


### EmbedDKT
```shell
# EmbedDKT
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --ctx="cuda:0" --model_name EmbedDKT --root=$HOME/TKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
# EmbedDKT+ 
python3 DKT.py train \$data_dir/train.json \$data_dir/test.json --hyper_params "nettype=EmbedDKT;ku_num=int(835);hidden_num=int(900);latent_dim=int(600);dropout=float(0.5)" --loss_params "lr=float(0.1);lw1=float(0.003);lw2=float(3.0)" --ctx="cuda:0" --model_name EmbedDKT+ --root=$HOME/TKT --root_data_dir=\$root/data/ktbd/\$dataset --data_dir=\$root_data_dir --dataset=junyi
```


## Appendix

### Model
There are a lot of models that implements different knowledge tracing models in different frameworks, 
the following are the url of those implemented by python (the stared is the authors version):

* DKT [[tensorflow]](https://github.com/mhagiwara/deep-knowledge-tracing)

* DKT+ [[tensorflow*]](https://github.com/ckyeungac/deep-knowledge-tracing-plus)

* DKVMN [[mxnet*]](https://github.com/jennyzhang0215/DKVMN)

* KTM [[libfm]](https://github.com/jilljenn/ktm)

* EKT[[pytorch*]](https://github.com/bigdata-ustc/ekt)

### Dataset
There are some datasets which are suitable for this task, 
you can refer to [BaseData ktbd doc](https://github.com/bigdata-ustc/EduData/blob/master/docs/ktbd.md) 
for these datasets 