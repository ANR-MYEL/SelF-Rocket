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

Hydra-SelF-Rocket is as accurate as Hydra-MultiRocket and HIVE-COTE v2.0, despite having less features, and also has a relatively shorter classifier prediction time.

### Comparison of SelF-Rocket and other SOTA methods performance (critical difference diagram and heatmap)

![](img/crit_diag_sr_hit.png)

![](img/heatmap_HS.png)

## Reproducing the results

### Data
The paper [Bake off redux: a review and experimental evaluation of recent time series classification algorithms](https://arxiv.org/abs/2304.13029) provide with [tsml-eval](https://tsml-eval.readthedocs.io/en/latest/publications/2023/tsc_bakeoff/tsc_bakeoff_2023.html) a folder containing all the 112 UCR datatets with their 30 resamples available [here](https://drive.google.com/file/d/1V36LSZLAK6FIYRfPx6mmE5euzogcXS83/view)

### Performance of each couple IR-PO (Section 3)

To obtain Table 1 of our paper, run [mod_minirocket](./code/performance_mod_minirocket.py). It will generate a csv file with the mean performance of each IR-PO couple for all the 112 UCR datasets selected with 30 resamples.

### Feature Generation (Section 4)

While SelF-Rocket uses [MiniRocket](https://arxiv.org/abs/2012.08791) as baseline, our code is based on the original implementation of [MultiRocket](https://arxiv.org/abs/2102.00457). 9996 or 19992 features are extracted from each combinaison (IR-PO). In this implementation IR = {BASE,DIFF}, PO = {PPV,GMP,MPV,MIPV,LSPV}. The source code of the Feature Generation module is available [here](./code/features_generator.py)

### Performance of SelF-Rocket (Section 5)

To obtain the data required for the critical difference diagram and the heatmap, for SelF-Rocket run    [main_ucr112_SR](./code/main_ucr112_SR.py) and for Hydra SelF-Rocket run [main_ucr112_HSR](./code/main_ucr112_HSR.py)

```
Arguments:
-df --inputDataFolder type=str : path of datasets (required)
-k --k_fold type=int : number of folds
-r --num_resamples type=int : number of resamples to use
-nr --num_runs type=int : number of runs to do

Examples:
> py main_ucr112_SR.py -df "../datasets_UCR_resamp_tsv/" -k 2 -nr 10
> py main_ucr112_HSR.py -df "../datasets_UCR_resamp_tsv/" -k 2 -nr 10
 
``` 
This will generate three files : a main file containing the mean accuracy over 30 resamples for each dataset (e.g. [Mean_perf_SR_k2_nr10](./results/Mean_perf_SR_k2_nr10.csv), [Mean_perf_HSR_k2_nr10](./results/Mean_perf_HSR_k2_nr10.csv)), a second file containing the accuracy for each resample for all datasets (e.g. [Perf_rsmpl_SR_k2_nr10](./results/Perf_rsmpl_SR_k2_nr10.csv), [Perf_rsmpl_HSR_k2_nr10](./results/Perf_rsmpl_HSR_k2_nr10.csv)) and a third file containing the selected IR-PO for each resample and dataset (e.g. [IR_PO_rsmpl_SR_k2_nr10](./results/IR_PO_rsmpl_SR_k2_nr10.csv), [IR_PO_rsmpl_HSR_k2_nr10](./results/IR_PO_rsmpl_HSR_k2_nr10.csv)).

The mean performance (over 30 resamples) of SelF-Rocket & Hydra SelF-Rocket (k = 2, f = 5000, nr = 10) on the 112 selected UCR datasets.

| dataset                        | SelF-Rocket MEAN ACCURACY | Hydra SelF-Rocket MEAN ACCURACY | MINIROCKET MEAN ACCURACY |
|--------------------------------|---------------------------|---------------------------------|--------------------------|
| ACSF1                          | 0.8253333333333334        | 0.839666666666667               | 0.8246666666666667       |
| Adiac                          | 0.8250639386189257        | 0.834441602728048               | 0.8015345268542199       |
| ArrowHead                      | 0.8849523809523808        | 0.881142857142857               | 0.8832380952380954       |
| BME                            | 0.9922222222222222        | 0.997777777777778               | 0.9919999999999998       |
| Beef                           | 0.7944444444444445        | 0.776666666666667               | 0.7655555555555554       |
| BeetleFly                      | 0.9149999999999999        | 0.955                           | 0.9099999999999999       |
| BirdChicken                    | 0.9033333333333331        | 0.911666666666666               | 0.9183333333333332       |
| CBF                            | 0.996074074074074         | 0.993740740740741               | 0.9965185185185186       |
| Car                            | 0.9233333333333333        | 0.924444444444445               | 0.9205555555555556       |
| Chinatown                      | 0.9679300291545191        | 0.966277939747328               | 0.9687074829931972       |
| ChlorineConcentration          | 0.7866840277777776        | 0.775321180555556               | 0.753532986111111        |
| CinCECGTorso                   | 0.9602657004830916        | 0.995096618357488               | 0.8745169082125603       |
| Coffee                         | 0.9988095238095238        | 0.996428571428572               | 0.9988095238095238       |
| Computers                      | 0.8420000000000001        | 0.8712                          | 0.8033333333333332       |
| CricketX                       | 0.8211111111111112        | 0.831196581196581               | 0.8241880341880342       |
| CricketY                       | 0.8375213675213675        | 0.85                            | 0.8399999999999999       |
| CricketZ                       | 0.8389743589743589        | 0.85025641025641                | 0.8391452991452992       |
| Crop                           | 0.766484126984127         | 0.770444444444444               | 0.7641011904761903       |
| DiatomSizeReduction            | 0.9506535947712416        | 0.943899782135076               | 0.9436819172113288       |
| DistalPhalanxOutlineAgeGroup   | 0.7983213429256595        | 0.801438848920863               | 0.7944844124700239       |
| DistalPhalanxOutlineCorrect    | 0.8380434782608694        | 0.838768115942029               | 0.8253623188405796       |
| DistalPhalanxTW                | 0.6999999999999998        | 0.698561151079137               | 0.6892086330935251       |
| ECG200                         | 0.8940000000000001        | 0.894333333333333               | 0.8990000000000002       |
| ECG5000                        | 0.9464962962962963        | 0.947281481481481               | 0.9464888888888888       |
| ECGFiveDays                    | 0.991637630662021         | 0.994928377855207               | 0.9905536198219127       |
| EOGHorizontalSignal            | 0.8575506445672192        | 0.862338858195212               | 0.8360957642725599       |
| EOGVerticalSignal              | 0.8085635359116022        | 0.813075506445672               | 0.7995395948434623       |
| Earthquakes                    | 0.7422062350119903        | 0.743165467625899               | 0.7333333333333332       |
| ElectricDevices                | 0.8918730817446936        | 0.901357368261791               | 0.8745774434790127       |
| EthanolLevel                   | 0.7108                    | 0.704533333333333               | 0.6635333333333334       |
| FaceAll                        | 0.9831755424063118        | 0.983668639053254               | 0.983293885601578        |
| FaceFour                       | 0.9397727272727271        | 0.911742424242424               | 0.9412878787878787       |
| FacesUCR                       | 0.9684552845528456        | 0.966032520325203               | 0.9686016260162602       |
| FiftyWords                     | 0.8313553113553115        | 0.838168498168498               | 0.8284981684981686       |
| Fish                           | 0.9809523809523811        | 0.98247619047619                | 0.9754285714285716       |
| FordA                          | 0.9495202020202019        | 0.956590909090909               | 0.9449747474747474       |
| FordB                          | 0.9241975308641975        | 0.930740740740741               | 0.9207818930041152       |
| FreezerRegularTrain            | 0.9992631578947369        | 0.998257309941521               | 0.9990877192982457       |
| FreezerSmallTrain              | 0.9885380116959066        | 0.982690058479532               | 0.9876023391812866       |
| GunPoint                       | 0.9955555555555554        | 0.998666666666667               | 0.9919999999999999       |
| GunPointAgeSpan                | 0.9938818565400844        | 0.994620253164557               | 0.9924050632911392       |
| GunPointMaleVersusFemale       | 1.0                       | 0.999894514767933               | 1.0                      |
| GunPointOldVersusYoung         | 1.0                       | 1                               | 1.0                      |
| Ham                            | 0.8431746031746032        | 0.852380952380953               | 0.8488888888888889       |
| HandOutlines                   | 0.9418018018018017        | 0.947387387387387               | 0.9382882882882881       |
| Haptics                        | 0.519047619047619         | 0.532142857142857               | 0.5452380952380953       |
| Herring                        | 0.6098958333333333        | 0.614583333333333               | 0.6088541666666667       |
| HouseTwenty                    | 0.9680672268907564        | 0.975350140056023               | 0.965546218487395        |
| InlineSkate                    | 0.5084848484848484        | 0.524484848484849               | 0.4896363636363636       |
| InsectEPGRegularTrain          | 1.0                       | 1                               | 1.0                      |
| InsectEPGSmallTrain            | 0.9908969210174028        | 0.996519410977242               | 0.993172690763052        |
| InsectWingbeatSound            | 0.6613973063973064        | 0.663451178451178               | 0.6575420875420875       |
| ItalyPowerDemand               | 0.9628117913832199        | 0.962099125364431               | 0.9602202785876254       |
| LargeKitchenAppliances         | 0.9262222222222222        | 0.939555555555556               | 0.9024000000000001       |
| Lightning2                     | 0.7459016393442623        | 0.773770491803279               | 0.7601092896174866       |
| Lightning7                     | 0.7990867579908674        | 0.810045662100457               | 0.7771689497716895       |
| Mallat                         | 0.9569012082444919        | 0.960653873489694               | 0.9551528073916135       |
| Meat                           | 0.9916666666666667        | 0.987222222222222               | 0.9933333333333335       |
| MedicalImages                  | 0.7982894736842107        | 0.80469298245614                | 0.8048245614035087       |
| MiddlePhalanxOutlineAgeGroup   | 0.6586580086580087        | 0.667965367965368               | 0.6502164502164501       |
| MiddlePhalanxOutlineCorrect    | 0.85028636884307          | 0.851202749140894               | 0.843413516609393        |
| MiddlePhalanxTW                | 0.5532467532467533        | 0.557575757575758               | 0.5541125541125542       |
| MixedShapesRegularTrain        | 0.975257731958763         | 0.981127147766323               | 0.9693745704467354       |
| MixedShapesSmallTrain          | 0.9487972508591067        | 0.957223367697594               | 0.9421030927835052       |
| MoteStrain                     | 0.9133919062832802        | 0.926730564430245               | 0.9136581469648563       |
| NonInvasiveFetalECGThorax1     | 0.9595250212044104        | 0.961984732824428               | 0.9518066157760815       |
| NonInvasiveFetalECGThorax2     | 0.966412213740458         | 0.971077184054283               | 0.9635114503816793       |
| OSULeaf                        | 0.965977961432507         | 0.978925619834711               | 0.9568870523415979       |
| OliveOil                       | 0.9088888888888889        | 0.917777777777778               | 0.9155555555555556       |
| PhalangesOutlinesCorrect       | 0.8513597513597513        | 0.852059052059052               | 0.8457653457653458       |
| Phoneme                        | 0.33973277074542896       | 0.352232770745429               | 0.2908403656821379       |
| PigAirwayPressure              | 0.9261217948717948        | 0.912980769230769               | 0.8748397435897437       |
| PigArtPressure                 | 0.9549679487179484        | 0.961057692307692               | 0.9596153846153848       |
| PigCVP                         | 0.9099358974358974        | 0.909455128205128               | 0.9179487179487179       |
| Plane                          | 1.0                       | 1                               | 1.0                      |
| PowerCons                      | 0.9829629629629629        | 0.970185185185185               | 0.9838888888888887       |
| ProximalPhalanxOutlineAgeGroup | 0.8471544715447157        | 0.845691056910569               | 0.8450406504065041       |
| ProximalPhalanxOutlineCorrect  | 0.9083619702176404        | 0.905727376861398               | 0.8961053837342499       |
| ProximalPhalanxTW              | 0.8014634146341464        | 0.809593495934959               | 0.8032520325203251       |
| RefrigerationDevices           | 0.7473777777777778        | 0.786933333333333               | 0.707822222222222        |
| Rock                           | 0.844                     | 0.874                           | 0.8046666666666666       |
| ScreenType                     | 0.6293333333333334        | 0.679111111111111               | 0.5584888888888889       |
| SemgHandGenderCh2              | 0.9526666666666667        | 0.935                           | 0.9108333333333334       |
| SemgHandMovementCh2            | 0.7919259259259263        | 0.733555555555556               | 0.6834074074074075       |
| SemgHandSubjectCh2             | 0.9304444444444446        | 0.911555555555556               | 0.8821481481481481       |
| ShapeletSim                    | 0.9994444444444444        | 0.992777777777778               | 0.9996296296296295       |
| ShapesAll                      | 0.9413333333333332        | 0.948666666666667               | 0.9349444444444444       |
| SmallKitchenAppliances         | 0.8251555555555554        | 0.8352                          | 0.8137777777777779       |
| SmoothSubspace                 | 0.9686666666666669        | 0.962444444444444               | 0.9586666666666667       |
| SonyAIBORobotSurface1          | 0.9539101497504163        | 0.947920133111481               | 0.9480865224625624       |
| SonyAIBORobotSurface2          | 0.9544596012591816        | 0.954214760405736               | 0.9415879678209162       |
| StarLightCurves                | 0.9816091954022989        | 0.981495871782419               | 0.9814351627003399       |
| Strawberry                     | 0.9804504504504504        | 0.980720720720721               | 0.9791891891891892       |
| SwedishLeaf                    | 0.9732799999999998        | 0.977386666666667               | 0.9615466666666668       |
| Symbols                        | 0.969715242881072         | 0.974070351758794               | 0.9670686767169179       |
| SyntheticControl               | 0.9918888888888887        | 0.989444444444445               | 0.9912222222222222       |
| ToeSegmentation1               | 0.9371345029239767        | 0.942982456140351               | 0.9410818713450291       |
| ToeSegmentation2               | 0.9464102564102563        | 0.949230769230769               | 0.9410256410256408       |
| Trace                          | 1.0                       | 1                               | 1.0                      |
| TwoLeadECG                     | 0.9973953760608723        | 0.998273339186421               | 0.9973075797483174       |
| TwoPatterns                    | 0.9997749999999999        | 0.998025                        | 0.994975                 |
| UMD                            | 0.9905092592592594        | 0.988888888888889               | 0.9905092592592594       |
| UWaveGestureLibraryAll         | 0.9815093988460823        | 0.982998324958124               | 0.9749860413176996       |
| UWaveGestureLibraryX           | 0.8636422855015818        | 0.871533593895403               | 0.8566815559277872       |
| UWaveGestureLibraryY           | 0.7972827098455237        | 0.806690861715987               | 0.7867857807556298       |
| UWaveGestureLibraryZ           | 0.8068397543271915        | 0.816582914572864               | 0.8015726782058441       |
| Wafer                          | 0.9998053212199871        | 0.999945922561108               | 0.9988860047588145       |
| Wine                           | 0.928395061728395         | 0.926543209876543               | 0.9290123456790124       |
| WordSynonyms                   | 0.7723615464994774        | 0.777115987460815               | 0.7682863113897596       |
| Worms                          | 0.7372294372294373        | 0.757142857142857               | 0.7320346320346319       |
| WormsTwoClass                  | 0.7809523809523812        | 0.791774891774892               | 0.7913419913419912       |
| Yoga                           | 0.9325333333333334        | 0.939133333333334               | 0.9190666666666666       |

## Usage of SelF-Rocket & Hydra-SelF-Rocket Classifier

```python
from self_rocket import SelFRocket

# The shape could be (num_examples,ts_length) or (num_examples,1,ts_length)

X_train, y_train = ...
X_test, y_test = ...

model = SelFRocket(num_runs = 10,num_kfold = 2)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
``` 

```python
from hydra_self_rocket import HydraSelFRocket

# The shape could be (num_examples,ts_length) or (num_examples,1,ts_length)

X_train, y_train = ...
X_test, y_test = ...

model = HydraSelFRocket(num_runs = 10,num_kfold = 2)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
``` 

## Acknowledgements

This research project, supported and financed by the ANR (Agence Nationale pour la Recherche), is part of the Labcom (Laboratoire Commun) MYEL (MobilitY and Reliability of Electrical chain Lab) involving LSEE, LGI2A and CRITTM2A.

