# Adding_HapkeCNN_to_HySUPP
This is a page to explain how to add the externel unmixing model to [HySUPP](https://github.com/inria-thoth/HySUPP) framework.

## How to
1. Set up HySUPP by following official guidance
2. Copy "HapkeCNN.py" to "\HySUPP-main\src\model\blind", "HapkeCNN.yaml" to "\HySUPP-main\config\model".
3. Add "from .HapkeCNN import HapkeCNN" in "\HySUPP-main\src\model\blind\__init__.py"

## Tips to use HySUPP
### Dataset Preparation
- Data format can be found in README of HySUPP.
- A config file for the own data set needs to be prepared. (You can copy any yaml file and modify for your dataset)
```
name: src.data.base.HSIWithGT　<-in the case of without GT, write "src.data.base.RealHSI" instead
dataset: "dataset　　＜ーfile name of dataset.mat
#p:16                    <-for in the case of "src.data.base.RealHSI"
data_dir: ${DATA_dir}
figs_dir: ${FIGS_dir}
```
### Parameter Tuning
- You can find names of parameters from their config yaml files.
- For grid search, you can assign multiple values for each parameter as follows;
```
python unmixing.py mode=semi data=DC1 model=SUnCNN projection=True model.niters=2000,4000,8000 model.noisy_input=False,True noise.SNR=30
```

### Matlab-based Packages
- Some of the models are implemented in Matlab.
- You can get those models from the link described in Table 1 of the HySUPP paper (or "\HySUPP-main\src\model\matlab.py").
- Once you get the Matlab code, you need to install matlabengine to your python environment, followed by adjusting the file directory ans modifying config.yml (matlab root).

## Tested Environment
### Hardware
- Processor	Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz (3.91 GHz)
- Installed RAM	64.0 GB (63.7 GB usable)
- Graphics card	NVIDIA RTX A4000 (16 GB)
- Edition	Windows 11 Pro for Workstations

### Software
- Python 3.9.23
