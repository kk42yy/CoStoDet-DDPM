# CoStoDet-DDPM
CoStoDet-DDPM: Collaborative Training of Stochastic and Deterministic Models Improves Surgical Workflow Anticipation and Recognition

## 1. Preparation

### Step 1:

<details>
<summary>Download the Cholec80, AutoLaparo</summary>

- Access can be requested [Cholec80](http://camma.u-strasbg.fr/datasets), [AutoLaparo](https://autolaparo.github.io/).
- Download the videos for each datasets and extract frames at 1fps. E.g. for `video01.mp4` with ffmpeg, run:
```bash
mkdir /<PATH_TO_THIS_FOLDER>/data/frames_1fps/01/
ffmpeg -hide_banner -i /<PATH_TO_VIDEOS>/video01.mp4 -r 1 -start_number 0 /<PATH_TO_THIS_FOLDER>/data/frames_1fps/01/%08d.jpg
```
- DACAT also prepare a shell file to extract at [here](https://github.com/kk42yy/DACAT/blob/main/src/video2img.sh)
- The final dataset structure should look like this:

```
Cholec80/
	data/
		frames_1fps/
			01/
				00000001.jpg
				00000002.jpg
				00000003.jpg
				00000004.jpg
				...
			02/
				...
			...
			80/
				...
		phase_annotations/
			video01-phase.txt
			video02-phase.txt
			...
			video80-phase.txt
		tool_annotations/
			video01-tool.txt
			video02-tool.txt
			...
			video80-tool.txt
	output/
	train_scripts/
	predict.sh
	train.sh
```

- When training the anticipation model with both tool and phase, please combine the `phase_annotations` and `tool_annotations`, we also have prepared [here](https://huggingface.co/kk42yy/CoStoDet-DDPM/blob/main/tool_phase_annotations.zip).

</details>

### Step 2: 

<details>
<summary>Download pretrained models ConvNeXt V1 and ConvNeXt V2-T</summary>

- download ConvNeXt-T [weights](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) and place here: `.../[DATASET]/train_scripts/convnext/convnext_tiny_1k_224_ema.pth` and `.../[DATASET]/train_scripts\newly_opt_ykx\LongShortNet\convnext/convnext_tiny_1k_224_ema.pth`
- download ConvNeXt V2-T [weights](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt) and place here: `.../[DATASET]/train_scripts/convnext/convnextv2_tiny_1k_224_ema.pt` and `.../[DATASET]/train_scripts\newly_opt_ykx\LongShortNet\convnext/convnextv2_tiny_1k_224_ema.pt`

</details>

### Step 3: 
<details>
<summary>Environment Requirements</summary>


See [requirements.txt](requirements.txt).

</details>


## 2. Train

### 2.1 Anticipation Training
```bash
source .../Cholec80/train_anti.sh
```

### 2.2 Recognition Training
#### 2.2.1 Training feature cache [Optional]
```bash
### BNPifalls (Training feature cache) for Cholec80
python3 train.py phase \
    --split cuhk4040 \
    --trial_name BNPitfall4040 \
    --backbone convnext --freeze --workers 4 --seq_len 256 --lr 1e-4

### BNPifalls (Training feature cache) for AutoLaparo
python3 train.py phase \
	--split cuhk1007 \
	--trial_name Step1 \
	--backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 5e-4
```

After training, please rename and save the checkpoint `.../output/checkpoints/phase/YourTrainNameXXX/models/checkpoint_best_acc.pth.tar` in `.../[DATASET]/train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar`.

#### We also released this stage models at [TrainingRequirement](https://huggingface.co/kk42yy/CoStoDet-DDPM/tree/main/Recognition/TrainingRequirement) for Cholec80 and AutoLaparo.

#### 2.2.2 Train CoStoDet-DDPM
```bash
### Cholec80
source .../Cholec80/train_DACAT.sh

### AutoLaparo
source .../AutoLaparo/train_DACAT.sh
```

## 3. Infer

### All the models for Anticipation and Rrecognition have been released at: [CoStoDet-DDPM](https://huggingface.co/kk42yy/CoStoDet-DDPM/tree/main)
Set the model path in `.../[DATASET]/predict.sh` and 
```bash
### Anticipation
source .../Cholec80/predict_anti.sh

### Recognition
source .../Cholec80/predict.sh
source .../AutoLaparo/predict.sh
```


## 4. Evaluate

### 4.1 Cholec80
Use the [Python file](Cholec80/train_scripts/newly_opt_ykx/evaluation_total.py/#L66).

### 4.2 AutoLaparo
Use the [Python file](AutoLaparo/train_scripts/newly_opt_ykx/evaluation_total.py/#L66).

## Reference
* [BNPitfalls (MIA 24)](https://gitlab.com/nct_tso_public/pitfalls_bn)
* [DACAT (ICASSP 25)](https://github.com/kk42yy/DACAT)

# Citations
If you find this repository useful, please consider citing our paper:
```
@article{yang2025costodet,
  title={CoStoDet-DDPM: Collaborative Training of Stochastic and Deterministic Models Improves Surgical Workflow Anticipation and Recognition},
  author={Yang, Kaixiang and Li, Xin and Li, Qiang and Wang, Zhiwei},
  journal={arXiv preprint arXiv:2503.10216},
  year={2025}
}
```
