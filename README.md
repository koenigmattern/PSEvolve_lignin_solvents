# "Machine-guided solvent design for lignin-first biorefineries and lignin upgrading" 

<img src="https://github.com/koenigmattern/PSEvolve_lignin_solvents/blob/main/auxil/PSEvolve_logo.png" width="300">

PSEvolve is a genetic algorithm for molecular design (available [here](https://github.com/koenigmattern/PSEvolve) In this repo, PSEvolve was tailored for designing
solvents with high lignin solubilities and for aldehyde-assisted fractionation.

main_lignin.py: Main file for solvent design. 
- use group_constraints = 'AAF' for solvent design for aldehyde-assisted fractionation
- use group_constraints = None for simply designin solvents with high lignin solubility

utils.py:
- coantains all functions of PSEvolve, tailored for lignin solvent design

get_GNN_pred.py
- if you simply want to predict the lignin solubilities of specific solvents, use this function
- specify the solvents in "pred_mols.csv"
- run this function and open "pred_mols.csv" again to see the results

How to cite PSEvolve:

```
@Article{XXX,
author ="Laura König-Mattern, Edgar I. Sancez Medina, Anastasia O. Komarova, Steffen Linke, Liisa Rihko-Struckmann, Jeremy Luterbacher, Kai Sundmacher",
title  ="Machine-guided solvent design for lignin-first biorefineries and lignin upgrading",
journal  ="XXX",
year  ="XXX",
volume  ="X",
issue  ="X",
pages  ="XX",
publisher  ="XXX",
doi  ="XXX",
url  ="XXX"}

```

## Requirements

The following packages are required:

- RDKit >= 2021.03.1
- networkx >= 2.6.3
- PyTorch >= 1.8.0
- Pytorch Geometric >= 2.0


## License 

The code is licensed under XXXXX