# Experimental Logs

The source code of our [latest arXiv paper](https://arxiv.org/abs/2412.04282v2) is released in this repo, please test it if you are interested. For the false setting problem in the Table 1 of our [first arXiv paper](https://arxiv.org/pdf/2412.04282v1), we illustrate the setting and experiment logs of every sequence. 


| Scene            | Cook Spinach | Sear Steak | Flame Steak | Flame Salmon | Coffee Martini | Cut Roast Beef |
|------------------|:------------:|:----------:|:-----------:|:------------:|:--------------:|:--------------:|
| training setting |    correct   |    wrong   |    wrong    |     wrong    |      wrong     |      wrong     |
| eval setting     |   eval=True  | eval=False |  eval=False |  eval=False  |   eval=False   |    eval=False  |
| results          |   PSNR=30.17 | PSNR=38.43 |  PSNR=37.64 |  PSNR=31.85  |   PSNR=32.52   |    PSNR=36.96  |
| file-time        |   2024.11.05 | 2024.11.05 | 2024.11.05  |  2024.11.04  |   2024.11.04   |    2024.11.05  |

