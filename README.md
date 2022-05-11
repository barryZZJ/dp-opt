# DP-Opt

An enhanced disprover algorithm which aims to resolve the limitations in counterexample construction of the latest work [DP-Sniper](https://github.com/eth-sri/dp-sniper) and produce higher privacy violations.

## Install

```bash
pip install -r requirements.txt
```

We used python version 3.8 to run our evaluations.

## Basic Usage

To run the evaluation, run the following command:

```bash
python dpopt/experiments/__main__.py
```

The main algorithms DP-Opt and PowerSearcher from the paper can be found in [dpopt/search/powersearcher.py](dpopt/search/powersearcher.py) and 

[dpopt/attack/dpopt.py](dpopt/attack/dpopt.py).


# Liscence

MIT License, see [LICENSE](LICENSE).

This repository is derived from [DP-Sniper](https://github.com/eth-sri/dp-sniper), marked as `MIT License, Copyright (c) 2021 SRI Lab, ETH Zurich`.