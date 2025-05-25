# EDTformer

## Getting Started
This repo follows the framework of [CricaVPR](https://github.com/Lu-Feng/CricaVPR), and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark). We utilize the GSV-Cities dataset for training and you can download it [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:

```
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```
Before training, you should download the pre-trained foundation model DINOv2(ViT-B/14) [HERE](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth).

## Train
```
python3 train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=pitts30k --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_pretrain.pth --epochs_num=15
```

## Test
To evaluate the trained model:
```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --resume=/path/to/trained/model/your_model.pth
```
## Trained Model
You can directly download the trained model [HERE](https://drive.google.com/file/d/1T7qmq1NtA8NgN8uLrrgsckxMLX_nWDJb/view?usp=sharing).

## Acknowledgements

Parts of this repo are inspired by the following repositories:

[CricaVPR](https://github.com/Lu-Feng/CricaVPR)

[Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

[GSV-Cities](https://github.com/amaralibey/gsv-cities)

## Citation
If you find this repo useful for your research, please consider leaving a star⭐️ and citing the paper
```
@ARTICLE{EDTformer,
  author={Jin, Tong and Lu, Feng and Hu, Shuyu and Yuan, Chun and Liu, Yunpeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={EDTformer: An Efficient Decoder Transformer for Visual Place Recognition}, 
  year={2025},
  doi={10.1109/TCSVT.2025.3559084}}}
```
