# Introduction of Experimental Logs 

The source code of our [latest arXiv paper](https://arxiv.org/abs/2412.04282v2) has been released in this repository. Please feel free to test it if you are interested.
For the false setting problem in the Table 1 of our [first arXiv paper](https://arxiv.org/pdf/2412.04282v1), we provide the detailed training settings and experimental logs for each sequence below:


| Scene            | Cook Spinach | Sear Steak | Flame Steak | Flame Salmon | Coffee Martini | Cut Roast Beef |
|------------------|:------------:|:----------:|:-----------:|:------------:|:--------------:|:--------------:|
| Training Setting |    correct   |    wrong   |    wrong    |     wrong    |      wrong     |      wrong     |
| Parameter Setting |   `eval=True`  | `eval=False` |  `eval=False` |  `eval=False`  |   `eval=False`   |    `eval=False`  |
| Timestamp of Files |   2024.11.05 | 2024.11.05 | 2024.11.05  |  2024.11.04  |   2024.11.04   |    2024.11.05  |
| Config Args      | [cfg_args](/experimental_logs/asset/cook_spinach_node/cfg_args) | [cfg_args](/experimental_logs/asset/sear_steak_node/cfg_args) | [cfg_args](/experimental_logs/asset/flame_steak_node/cfg_args) | [cfg_args](/experimental_logs/asset/flame_salmon_1_node/cfg_args) | [cfg_args](/experimental_logs/asset/coffee_martini_node/cfg_args) | [cfg_args](/experimental_logs/asset/cut_roasted_beef_node/cfg_args) |

✅ Notes:
You can obtain the timestamp of each file via following operations:

- **On Ubuntu**:
```
cd asset/cook_spinach_node
ls -l --time-style=long-iso
```
where the date is the timestamp.

- **On Windows**: 
just right-click → Properties → check "Modified" time.