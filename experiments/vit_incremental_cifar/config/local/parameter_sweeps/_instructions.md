# Instructions on how to run ViT Incremental CIFAR-100 experiment locally

I use the `mlproject-manager` package to run systematic experiments locally. 
Follow these steps to run the experiments:

1. Export python path and load the virtual environment
```bash
    export PYTHONPATH=.
    source ./venv/bin/activate
```

2. Register the experiment using the following command:
```bash
python -m mlproj_manager.experiments.register_experiment \
--experiment-path ./experiments/vit_incremental_cifar/vit_experiment.py  \
--experiment-class-name IncrementalCIFARExperiment \
--experiment-name _ 
````
replace `_` with the name you want to use for the experiment.

3. Run experiment
```bash
    python -m mlproj_manager.main --experiment-name _ --verbose \
    --experiment-config-path ./experiments/vit_incremental_cifar/config/local/parameter_sweeps/config  
```
replace `_` with the experiment name and `config` with the file name in the directory.
