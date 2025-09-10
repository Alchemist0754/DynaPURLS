# DynaPURLS

## TODOs
- Docs for:
  - Prerequisites
  - Demo
  - Data Preparation
  - Testing Pre-trained Models
  - Training
  - Citation
- Codes for:
  - Basic organization and transplantation from implemented codes
  - Pre-trained model data
  - Data preprocess

## Prerequisites
- Installation
- Get pretrained models
- Demo
- Data Preparation
- Testing Pretrained Models
- Training

## Training
`main.py` supports training a new model with customized configs. The script accepts the following parameters:

### Arguments

| Argument        | Possible Values                              | Description                                                                 |
|-----------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| `ntu`           | `60`; `120`                                   | Which NTU dataset to use                                                    |
| `ss`            | `5`; `12` (for NTU-60); `24` (for NTU-120)    | Which split to use                                                          |
| `st`            | `r` (random)                                  | Split type                                                                  |
| `phase`         | `train`; `val`                                | `train` (required for ZSL); run once with `train` and once with `val` for GZSL |
| `ve`            | `shift`; `msg3d`                              | Select the Visual Embedding model                                           |
| `le`            | `w2v`; `bert`                                 | Select the Language Embedding model                                         |
| `num_cycles`    | Integer                                        | Number of cycles (e.g., train for 10 cycles)                                |
| `num_epoch_per_cycle` | Integer                                 | Number of epochs per cycle (1700 for 5-random and 1900 for others)          |
| `latent_size`   | Integer                                        | Size of the skeleton latent dimension (100 for NTU-60, 200 for NTU-120)     |
| `load_epoch`    | Integer                                        | The epoch to load                                                           |
| `load_classifier` | —                                           | Set if the pre-trained classifier is to be loaded                           |
| `dataset`       | Path                                           | Path to the generated visual features                                       |
| `wdir`          | Path                                           | Directory to store the weights                                              |
| `mode`          | `train`; `eval`                               | `train` for training SYNSE, `eval` to evaluate using a pretrained model     |
| `gpu`           | Integer / ID                                   | GPU device number to train on                                               |

### Example
- To train PURLS for ZSL under a 55/5 split on NTU-60:
  - Command:  
    ```
    python main.py -c configs/5r.yml
    ```
- To test DynaPURLS for ZSL under a 55/5 split on NTU-60:
 - Command:  
    ```
    python test.py -c configs/5r.yml
    ```

## Citation
TBD

## Contact

For any questions, please feel free to create an issue or contact:

  * **Jingmin Zhu**: jingmin.zhu1@monash.edu
  * **Anqi Zhu**: maggie.zhu@monash.edu
  * **Qiuhong Ke**: Qiuhong.Ke@monash.edu
