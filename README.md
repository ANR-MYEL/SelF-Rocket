# SelF-Rocket


This repository contains the code related to the paper      

**Time series classification with random convolution kernels based transforms: pooling operators and input representations matter**

*Preprint*: [arxiv:2409.01115](https://arxiv.org/pdf/2409.01115)

> <div align="justify">This article presents a new approach based on MiniRocket, called SelF-Rocket, for fast time series classification (TSC). Unlike existing approaches based on random convolution kernels, it dynamically selects the best couple of input representations and pooling operator during the training process. SelF-Rocket achieves state-of-the-art accuracy on the University of California Riverside (UCR) TSC benchmark datasets.</div>

## Reference
Please cite:
```
@misc{lo2024timeseriesclassificationrandom,
      title={Time series classification with random convolution kernels based transforms: pooling operators and input representations matter}, 
      author={Mouhamadou Mansour Lo and Gildas Morvan and Mathieu Rossi and Fabrice Morganti and David Mercier},
      year={2024},
      eprint={2409.01115},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.01115}, 
}
```

## Results

**Note:  Results initially presented were incorrect.**

Hydra-SelF-Rocket is as accurate as Hydra-MultiRocket and HIVE-COTE v2.0, despite having less features, and also has a relatively shorter classifier training time.

### Comparison of SelF-Rocket and other SOTA methods performance (critical difference diagram and heat map)

![](img/crit_diag_sr_hit.png)

![](img/heatmap_HS.png)




