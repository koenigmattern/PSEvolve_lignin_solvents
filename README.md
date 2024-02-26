# Machine learning-supported solvent design for lignin-first biorefineries and lignin upgrading

<p align="center">
<img src="https://github.com/koenigmattern/PSEvolve_lignin_solvents/blob/main/auxil/PSEvolve_logo.png" width="300">
</p>

PSEvolve is a genetic algorithm for molecular design (available [here](https://github.com/koenigmattern/PSEvolve)). <br />
In this repo, PSEvolve was tailored for designing solvents with high lignin solubilities and for aldehyde-assisted fractionation.

main_lignin.py: Main file for solvent design. 
- use group_constraints = 'AAF' for solvent design for aldehyde-assisted fractionation
- use group_constraints = None for simply designin solvents with high lignin solubility

utils.py:
- coantains all functions of PSEvolve, tailored for lignin solvent design

get_GNN_pred.py
- if you simply want to predict the lignin solubilities of specific solvents, use this function
- specify the solvents in "pred_mols.csv"
- run this function and open "pred_mols.csv" again to see the results


How to cite this material:

```
@Article{
author ="Laura König-Mattern, Edgar I. Sancez Medina, Anastasia O. Komarova, Steffen Linke, Liisa Rihko-Struckmann, Jeremy Luterbacher, Kai Sundmacher",
title  ="Machine learning-supported solvent design for lignin-first biorefineries and lignin upgrading",
journal  ="Nature Chemical Engineering (submitted)",
year  ="2024"
}
```

## Requirements

The following packages are required:

- RDKit >= 2021.03.1
- networkx >= 2.6.3
- PyTorch >= 1.8.0
- Pytorch Geometric >= 2.0


## License 
The [dataset](Trained_GNN_LC/data/butina_L.csv) containing ca. 3300 lignin solubilities was generated using COSMO-RS (COSMthermX19, BIOVIA 3Ds). <br />
The dataset and the [GNN](Trained_GNN_LC) are licensed under [CC-BY-NC-SA 4.0](Trained_GNN_LC/LICENSE). <br />
PSEvolve is licensed under the [MIT License](LICENSE) and is free and provided as-is. <br />
If you use any code provided in this repository please cite the original publication.