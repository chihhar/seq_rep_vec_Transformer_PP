This repository provides data and codes used in the paper <em>A novel Transformer-based fully trainable point process</em> to Neuralcomputation.
# Contents
1. Instlation
2. Structure of folders and files
3. Citation
# Requirement
+ Python 3.7.13
+ Pytorch 1.11.0
# Installation
To get started with the project, follow these steps:
1. Clone the repository.
```
$ git clone https://github.com/chihhar/seq_rep_vec_Transformer_PP.git
```

2.  Install the required packages from the requirements.txt file using conda:
```
$ pip install -r requirements.txt
```
3. Download a zip file for each dataset, i.e., NCEDC and SF Police Call, from [Google Drive]   (https://drive.google.com/drive/folders/1bDROZjdKLxUslbnUY7q0JbxQZhG-KSin?usp=drive_link) and place unzipped files to the corresponding data folder under each dataset folder as the following structure.

4. Execute Main.py of each dataset to train and evaluate models as follows:
```
(training):$ python Main.py [--train ] [-gene]
(evaluation):$ python Main.py [-gene]
```
Option:
- train :
  - trainモード, 訓練時:True, 訓練済みモデルの検証:False, default:False
- gene
    - string: データセットの種類, default=h1
      - hawkes1 : h1
      - hawkes2 : h_fix05
      - seismic datasets (NCEDC) : jisin
      - SF police call datasets : 911_x_Address 
- 　trainvec_num:
  - int: seq_rep vecの数, default=3
- 　pooling_k:
  - int: anchor vecの数, default=3

5. Exececute mse_all_ttest.py of each dataset to compare the performance of multiple models :
    ```
    $ python mse_all_ttest.py
    ```
# Structure of folders and files
```
.
├── Main.py
├── Utils.py
├── checkpoint
├── kmeans.py
├── requirements.txt
├── data
│   ├── Address
│   ├── call1
│   ├── call2
│   ├── call3
│   ├── data_selector.py
│   ├── date_jisin.90016
│   ├── date_pickled.py
│   ├── h1
│   └── h_fix05
├── mse_all_ttest.py
├── pickled
├── related_works
│   ├── FT_PP
│   └── THP
├── ronbun_plot_code.py
└── transformer
    ├── Constants.py
    ├── Layers.py
    ├── Models.py
    ├── Modules.py
    ├── SubLayers.py
    └── __pycache__
```
# Seismic event data
We used seismic event data provided by the Northern California Earthquake Data Center (NCEDC (2014)) and extracted 18,470 seismic events with a magnitude of larger than 3.0 recorded between 1970 and January 2022. The elapsed time τ is recorded in hours.
|| # of train | # of validation | # of test |
|---| ---- | ---- | ---- |
|# of events | 14,776   | 1,847         |  1,847  |
|# of windows | 14,746   | 1,817         |  1,817  |

To train or valid seq_rep_vec_Transformer_PP model, execute Main.py as follows :
```python
(train): $ python Main.py -gene=jisin --train
(valid): $ python Main.py -gene=jisin
```

# Police call event data
We used police call event data provided by the City and County of
San Francisco (San-Francisco (2021)). We extracted 55,262 events occurring at the first, second, and third most frequently recorded addresses, that is, 800 Block of BRYANT ST, 800 Block of MARKET ST, and 1000 Block of POTRERO AV, during 2003–2018 and created sliding windows for each address. For stratification, we selected windows whose last event τn+1 is recorded between 3:00 PM and 1:00 AM. The elapsed time is recorded in hours.
|| # of train | # of validation | # of test |
|---| ---- | ---- | ---- |
|# of events | 44,228 | 5,510 | 5,524  |
|# of windows |  24,314 | 3,006 | 2,965 |

To train or valid seq_rep_vec_Transformer_PP model, execute Main.py as follows :
```
first data
(train):$ python Main.py -gene=911_1_Address --train
(valid):$ python Main.py -gene=911_1_Address

second data
(train):$ python Main.py -gene=911_2_Address --train
(valid):$ python Main.py -gene=911_2_Address

third data
(train):$ python Main.py -gene=911_3_Address --train
(valid):$ python Main.py -gene=911_3_Address
```

# update plot data and make fig.X
訓練済みモデルを読み込み、fig.4 ~ fig.8 のplotに必要なデータをpickle化する。(引数は Main.py を参照。)
```
$ python plot_code.py [-gene]
```
pickle化したデータを参照して、fig.4 ~ fig.8 までの図を作成。
```
$ python ronbun_plot_code.py
```
# Citation
