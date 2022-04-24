# Ensemble Brain Segmentation Model
 An ensemble of state of the art brain segmentation model to produce high accuraccy across all three kinds of tumour.

## Data Preparation:

Download the BraTS dataset.
The structure of the dataset should be as follows:

    dataset/
        - BraTS2021_AAAAA/
            BraTS2021_AAAAA_flair.nii.gz
            BraTS2021_AAAAA_t1.nii.gz
            BraTS2021_AAAAA_t1ce.nii.gz
            BraTS2021_AAAAA_t2.nii.gz
            BraTS2021_AAAAA_seg.nii.gz
        - BraTS2021_AAAAB/
            BraTS2021_AAAAB_flair.nii.gz
            BraTS2021_AAAAB_t1.nii.gz
            BraTS2021_AAAAB_t1ce.nii.gz
            BraTS2021_AAAAB_t2.nii.gz
            BraTS2021_AAAAB_seg.nii.gz
        ...
(The above uses nomenclature from BraTS 2021 dataset for the sake of demonstration.
The dataset directory should have a _folder_ for each subject, where each folder has files with nomenclature `{foldername}_{modality}.nii.gz`.


Preprocess the dataset as follows:
```shell
python data_preprocess.py --data_dir "path_to_dataset"
```

**[For training datasets only]:** Crop and save data as `.npy` files via:
```shell
python data_crop_npy.py --src_folder "path_to_src_folder" --dst_folder "path_to_dst_folder"
```

Perform train/val split on the training dataset as needed.
Place the newly generated splits in different folders, as the network's training session differentiates
between training and validation splits according to where it is placed (defined in `config.yaml` file).


## Neural Network Training/Testing
Training/Testing sessions are guided by the configuration defined in the `config.yaml` file.

### Anaconda

If you have [Anaconda](https://docs.anaconda.com/anaconda/install/) set-up on your system, install
[PyTorch](https://pytorch.org/) (tested with `1.8.1`, CUDA `11.1`).
Install the rest of the dependencies with:
```shell
pip install -r requirements.txt
```

Move to the `main` folder to execute the following commands.

**Training:**
```shell
python train.py --config config.yaml --gpu 0
```

**Testing:**
```shell
python test.py --config config.yaml --gpu 0
```

**[Note]:** The paths set internally in the docker container should match those provided in `config.yaml`, as those will
only be visible to the training/testing session.