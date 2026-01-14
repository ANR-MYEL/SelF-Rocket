# SelF-Rocket


This repository contains the code related to the paper      

**Time series classification with random convolution kernels: pooling operators and input representations matter**

*Preprint*: [arxiv:2409.01115](https://arxiv.org/abs/2409.01115v3)

> <div align="justify">This article presents a new approach based on MiniRocket, called SelF-Rocket, for fast time series classification (TSC). Unlike existing approaches based on random convolution kernels, it dynamically selects the best couple of input representations and pooling operator during the training process. SelF-Rocket achieves state-of-the-art accuracy on the University of California Riverside (UCR) TSC benchmark datasets.</div>

## Reference
Please cite:
```
@misc{lo2025timeseriesclassificationrandom,
      title={Time series classification with random convolution kernels: pooling operators and input representations matter}, 
      author={Mouhamadou Mansour Lo and Gildas Morvan and Mathieu Rossi and Fabrice Morganti and David Mercier},
      year={2025},
      eprint={2409.01115},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.01115}, 
}
```

## Results

Hydra + SelF-Rocket is as accurate as Hydra + MultiRocket and HIVE-COTE v2.0, despite having less features, and also has a relatively shorter classifier prediction time.

### Comparison of SelF-Rocket and other SOTA methods performance (critical difference diagram and heatmap)

![](img/Crit_diagr_HS.png)

![](img/heatmap_HS_fr.png)

## Reproducing the results

### Data
The paper [Bake off redux: a review and experimental evaluation of recent time series classification algorithms](https://arxiv.org/abs/2304.13029) provide with [tsml-eval](https://tsml-eval.readthedocs.io/en/latest/publications/2023/tsc_bakeoff/tsc_bakeoff_2023.html) a folder containing all the 112 UCR datatets with their 30 resamples available [here](https://drive.google.com/file/d/1V36LSZLAK6FIYRfPx6mmE5euzogcXS83/view)

### Performance of each couple IR-PO (Section 3)

To obtain Table 1 of our paper, run [mod_minirocket](./code/performance_mod_minirocket.py). It will generate a csv file with the mean performance of each IR-PO couple for all the 112 UCR datasets selected with 30 resamples.

### Feature Generation (Section 4)

While SelF-Rocket uses [MiniRocket](https://arxiv.org/abs/2012.08791) as baseline, our code is based on the original implementation of [MultiRocket](https://arxiv.org/abs/2102.00457). 9996 or 19992 features are extracted from each combinaison (IR-PO). In this implementation IR = {BASE,DIFF}, PO = {PPV,ZC,MPV,MIPV,LSPV}. The source code of the Feature Generation module is available [here](./code/features_generator.py)

### Performance of SelF-Rocket (Section 5)

To obtain the data required for the critical difference diagram and the heatmap, for SelF-Rocket run    [main_ucr112_SR](./code/main_ucr112_SR.py) and for Hydra SelF-Rocket run [main_ucr112_HSR](./code/main_ucr112_HSR.py)

```
Arguments:
-df --inputDataFolder type=str : path of datasets (required)
-fpc --num_features_pc type=int : number of features to use per mini-classifier
-r --num_resamples type=int : number of resamples to use
-nr --num_runs type=int : number of runs to do
-ir --only_mix type=bool : only use the MIX representation as Input Representation

Examples:
> py main_ucr112_SR.py -df "../datasets_UCR_resamp_tsv/" -fpc 5000 -nr 10 -ir
> py main_ucr112_HSR.py -df "../datasets_UCR_resamp_tsv/" -fpc 5000 -nr 10
 
``` 
This will generate three files : a main file containing the mean accuracy over 30 resamples for each dataset (e.g. [Mean_perf_SR](./results/Mean_perf_SR_UCR112_fpc5000_nr10.csv), [Mean_perf_HSR](./results/Mean_perf_HSR_UCR112_fpc5000_nr10.csv)), a second file containing the accuracy for each resample for all datasets (e.g. [Perf_rsmpl_SR](./results/Perf_rsmpl_SR_UCR112_fpc5000_nr10.csv), [Perf_rsmpl_HSR](./results/Perf_rsmpl_HSR_UCR112_fpc5000_nr10.csv)) and a third file containing the selected IR-PO for each resample and dataset (e.g. [IR_PO_rsmpl_SR](./results/IR_PO_rsmpl_SR_UCR112_fpc5000_nr10.csv), [IR_PO_rsmpl_HSR](./results/IR_PO_rsmpl_HSR_UCR112_fpc5000_nr10.csv)).

The mean performance (over 30 resamples) of SelF-Rocket & Hydra SelF-Rocket (f = 5000, nr = 10) on the 112 selected UCR datasets.

| dataset                        | SelF-Rocket MEAN ACCURACY | Hydra SelF-Rocket MEAN ACCURACY | MiniRocket MEAN ACCURACY |
|--------------------------------|---------------------------|---------------------------------|--------------------------|
| ACSF1                          | 0.827                     | 0.835666666666666               | 0.824666666666667        |
| Adiac                          | 0.826001705029838         | 0.832225063938619               | 0.80153452685422         |
| ArrowHead                      | 0.886857142857143         | 0.881333333333333               | 0.883238095238095        |
| Beef                           | 0.792222222222222         | 0.774444444444445               | 0.992                    |
| BeetleFly                      | 0.91                      | 0.95                            | 0.765555555555556        |
| BirdChicken                    | 0.901666666666667         | 0.906666666666667               | 0.91                     |
| BME                            | 0.991777777777778         | 0.998222222222222               | 0.918333333333333        |
| Car                            | 0.915                     | 0.911111111111111               | 0.996518518518519        |
| CBF                            | 0.996296296296296         | 0.994407407407407               | 0.920555555555556        |
| Chinatown                      | 0.966569484936832         | 0.966763848396501               | 0.968707482993197        |
| ChlorineConcentration          | 0.783984375               | 0.774314236111111               | 0.753532986111111        |
| CinCECGTorso                   | 0.971932367149759         | 0.994178743961353               | 0.874516908212561        |
| Coffee                         | 0.998809523809524         | 0.997619047619048               | 0.998809523809524        |
| Computers                      | 0.848533333333333         | 0.863333333333334               | 0.803333333333333        |
| CricketX                       | 0.818547008547009         | 0.83                            | 0.824188034188034        |
| CricketY                       | 0.833846153846154         | 0.846410256410256               | 0.84                     |
| CricketZ                       | 0.838461538461539         | 0.845384615384615               | 0.839145299145299        |
| Crop                           | 0.766926587301587         | 0.77015873015873                | 0.76410119047619         |
| DiatomSizeReduction            | 0.948148148148148         | 0.942156862745098               | 0.943681917211329        |
| DistalPhalanxOutlineAgeGroup   | 0.796642685851319         | 0.804076738609113               | 0.794484412470024        |
| DistalPhalanxOutlineCorrect    | 0.839251207729469         | 0.839130434782609               | 0.82536231884058         |
| DistalPhalanxTW                | 0.701918465227818         | 0.702158273381295               | 0.689208633093525        |
| Earthquakes                    | 0.742446043165468         | 0.744364508393286               | 0.899                    |
| ECG200                         | 0.893666666666667         | 0.897666666666667               | 0.946488888888889        |
| ECG5000                        | 0.946259259259259         | 0.947681481481482               | 0.990553619821913        |
| ECGFiveDays                    | 0.990669763840496         | 0.994696089818041               | 0.83609576427256         |
| ElectricDevices                | 0.891380279254744         | 0.900527385120823               | 0.799539594843462        |
| EOGHorizontalSignal            | 0.854788213627993         | 0.856261510128913               | 0.733333333333333        |
| EOGVerticalSignal              | 0.810405156537753         | 0.805248618784531               | 0.874577443479013        |
| EthanolLevel                   | 0.706666666666667         | 0.704133333333333               | 0.663533333333333        |
| FaceAll                        | 0.983550295857988         | 0.984142011834319               | 0.983293885601578        |
| FaceFour                       | 0.939015151515151         | 0.914015151515152               | 0.941287878787879        |
| FacesUCR                       | 0.968048780487805         | 0.966162601626016               | 0.96860162601626         |
| FiftyWords                     | 0.831135531135531         | 0.839267399267399               | 0.828498168498169        |
| Fish                           | 0.978857142857143         | 0.981523809523809               | 0.975428571428571        |
| FordA                          | 0.966186868686869         | 0.957979797979798               | 0.944974747474747        |
| FordB                          | 0.926666666666667         | 0.932181069958848               | 0.920781893004115        |
| FreezerRegularTrain            | 0.999157894736842         | 0.998081871345029               | 0.999087719298246        |
| FreezerSmallTrain              | 0.987929824561404         | 0.982631578947368               | 0.987602339181287        |
| GunPoint                       | 0.993777777777778         | 0.998222222222222               | 0.992                    |
| GunPointAgeSpan                | 0.993037974683544         | 0.995042194092827               | 0.992405063291139        |
| GunPointMaleVersusFemale       | 0.999894514767933         | 0.999894514767933               | 1                        |
| GunPointOldVersusYoung         | 1                         | 1                               | 1                        |
| Ham                            | 0.848571428571429         | 0.848571428571429               | 0.848888888888889        |
| HandOutlines                   | 0.938108108108108         | 0.943333333333333               | 0.938288288288288        |
| Haptics                        | 0.510930735930736         | 0.517532467532468               | 0.545238095238095        |
| Herring                        | 0.605729166666667         | 0.618229166666667               | 0.608854166666667        |
| HouseTwenty                    | 0.971148459383754         | 0.970588235294118               | 0.965546218487395        |
| InlineSkate                    | 0.502060606060606         | 0.53830303030303                | 0.489636363636364        |
| InsectEPGRegularTrain          | 1                         | 1                               | 1                        |
| InsectEPGSmallTrain            | 0.987014725568943         | 0.996251673360107               | 0.993172690763052        |
| InsectWingbeatSound            | 0.660808080808081         | 0.665252525252525               | 0.657542087542088        |
| ItalyPowerDemand               | 0.963654033041788         | 0.961451247165533               | 0.960220278587626        |
| LargeKitchenAppliances         | 0.924177777777778         | 0.938577777777778               | 0.9024                   |
| Lightning2                     | 0.73224043715847          | 0.75792349726776                | 0.760109289617486        |
| Lightning7                     | 0.794063926940639         | 0.802739726027397               | 0.77716894977169         |
| Mallat                         | 0.981250888415068         | 0.973987206823028               | 0.955152807391613        |
| Meat                           | 0.992777777777778         | 0.987777777777778               | 0.993333333333333        |
| MedicalImages                  | 0.799429824561404         | 0.807412280701754               | 0.804824561403509        |
| MiddlePhalanxOutlineAgeGroup   | 0.655194805194805         | 0.669047619047619               | 0.65021645021645         |
| MiddlePhalanxOutlineCorrect    | 0.848797250859107         | 0.848453608247423               | 0.843413516609393        |
| MiddlePhalanxTW                | 0.551731601731602         | 0.556277056277056               | 0.554112554112554        |
| MixedShapesRegularTrain        | 0.977951890034364         | 0.983628865979382               | 0.969374570446735        |
| MixedShapesSmallTrain          | 0.953676975945017         | 0.958927835051546               | 0.942103092783505        |
| MoteStrain                     | 0.9080404685836           | 0.926384451544196               | 0.913658146964856        |
| NonInvasiveFetalECGThorax1     | 0.960220525869381         | 0.960542832909245               | 0.951806615776082        |
| NonInvasiveFetalECGThorax2     | 0.965156912637829         | 0.969397794741306               | 0.963511450381679        |
| OliveOil                       | 0.906666666666667         | 0.91                            | 0.956887052341598        |
| OSULeaf                        | 0.971487603305785         | 0.979752066115702               | 0.915555555555556        |
| PhalangesOutlinesCorrect       | 0.851282051282051         | 0.853574203574204               | 0.845765345765346        |
| Phoneme                        | 0.368706047819972         | 0.356381856540084               | 0.290840365682138        |
| PigAirwayPressure              | 0.962980769230769         | 0.888461538461538               | 0.874839743589744        |
| PigArtPressure                 | 0.965224358974359         | 0.955929487179487               | 0.959615384615385        |
| PigCVP                         | 0.921794871794872         | 0.919391025641026               | 0.917948717948718        |
| Plane                          | 1                         | 1                               | 1                        |
| PowerCons                      | 0.981851851851852         | 0.971481481481481               | 0.983888888888889        |
| ProximalPhalanxOutlineAgeGroup | 0.845853658536585         | 0.847154471544716               | 0.845040650406504        |
| ProximalPhalanxOutlineCorrect  | 0.90950744558992          | 0.907789232531501               | 0.89610538373425         |
| ProximalPhalanxTW              | 0.803739837398374         | 0.808780487804878               | 0.803252032520325        |
| RefrigerationDevices           | 0.783555555555555         | 0.799644444444444               | 0.707822222222222        |
| Rock                           | 0.832666666666667         | 0.871333333333333               | 0.804666666666667        |
| ScreenType                     | 0.671644444444445         | 0.666844444444445               | 0.558488888888889        |
| SemgHandGenderCh2              | 0.953555555555556         | 0.934222222222222               | 0.910833333333333        |
| SemgHandMovementCh2            | 0.793111111111111         | 0.734518518518519               | 0.683407407407408        |
| SemgHandSubjectCh2             | 0.930074074074074         | 0.910666666666667               | 0.882148148148148        |
| ShapeletSim                    | 0.99962962962963          | 0.993148148148148               | 0.99962962962963         |
| ShapesAll                      | 0.940333333333334         | 0.948                           | 0.934944444444444        |
| SmallKitchenAppliances         | 0.825688888888889         | 0.838222222222222               | 0.813777777777778        |
| SmoothSubspace                 | 0.969111111111111         | 0.963333333333333               | 0.958666666666667        |
| SonyAIBORobotSurface1          | 0.948641153632834         | 0.947642817526345               | 0.948086522462562        |
| SonyAIBORobotSurface2          | 0.952535851696397         | 0.952430919902064               | 0.941587967820917        |
| StarLightCurves                | 0.981342075441153         | 0.981414926339647               | 0.98143516270034         |
| Strawberry                     | 0.98009009009009          | 0.981081081081081               | 0.979189189189189        |
| SwedishLeaf                    | 0.97184                   | 0.97424                         | 0.961546666666667        |
| Symbols                        | 0.971189279731993         | 0.973835845896148               | 0.967068676716918        |
| SyntheticControl               | 0.994444444444444         | 0.994555555555555               | 0.991222222222222        |
| ToeSegmentation1               | 0.940058479532164         | 0.942543859649123               | 0.941081871345029        |
| ToeSegmentation2               | 0.941025641025641         | 0.949487179487179               | 0.941025641025641        |
| Trace                          | 1                         | 1                               | 1                        |
| TwoLeadECG                     | 0.997073456248171         | 0.998507462686567               | 0.997307579748317        |
| TwoPatterns                    | 0.999991666666667         | 0.999625                        | 0.994975                 |
| UMD                            | 0.990509259259259         | 0.988657407407408               | 0.990509259259259        |
| UWaveGestureLibraryAll         | 0.981351200446678         | 0.981825795644891               | 0.9749860413177          |
| UWaveGestureLibraryX           | 0.864703145356412         | 0.872436255350828               | 0.856681555927787        |
| UWaveGestureLibraryY           | 0.798762330169365         | 0.806653638563186               | 0.78678578075563         |
| UWaveGestureLibraryZ           | 0.808738135120045         | 0.814386748557603               | 0.801572678205844        |
| Wafer                          | 0.999789097988319         | 0.999956738048886               | 0.998886004758815        |
| Wine                           | 0.929012345679012         | 0.930246913580247               | 0.929012345679012        |
| WordSynonyms                   | 0.769435736677116         | 0.777377220480669               | 0.76828631138976         |
| Worms                          | 0.765800865800866         | 0.769264069264069               | 0.732034632034632        |
| WormsTwoClass                  | 0.790909090909091         | 0.7995670995671                 | 0.791341991341991        |
| Yoga                           | 0.9318                    | 0.935544444444444               | 0.919066666666667        |



## Usage of SelF-Rocket & Hydra-SelF-Rocket Classifier

```python
from self_rocket import SelFRocket

# The shape could be (num_examples,ts_length) or (num_examples,1,ts_length)

X_train, y_train = ...
X_test, y_test = ...

model = SelFRocket(num_runs = 10, num_features_pc = 5000, only_MIX = False, num_kernels = 10000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
``` 

```python
from hydra_self_rocket import HydraSelFRocket

# The shape could be (num_examples,ts_length) or (num_examples,1,ts_length)

X_train, y_train = ...
X_test, y_test = ...

model = HydraSelFRocket(num_runs = 10, num_features_pc = 5000, only_MIX = False, num_kernels = 10000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
``` 

## Acknowledgements

This research project, supported and financed by the ANR (Agence Nationale pour la Recherche), is part of the Labcom (Laboratoire Commun) MYEL (MobilitY and Reliability of Electrical chain Lab) involving LSEE, LGI2A and CRITTM2A.

