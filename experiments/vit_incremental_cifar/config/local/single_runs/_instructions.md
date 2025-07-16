# Instructions on how to run ViT Incremental CIFAR-100 experiment locally

1. Export python path and load the virtual environment
```bash
export PYTHONPATH=.
source ./venv/bin/activate
```

2. Run the following command
```bash
    python ./experiments/vit_incremental_cifar/vit_experiment.py --verbose --gpu_index 0 --run_index 0 \
    --config_file ./experiments/vit_incremental_cifar/config/single_runs/config
````
replace `config` with the config you want to use.

3. If you want to run several indices, use the command below:
```bash
for INDEX in {0..4}; do
  echo "Running index $INDEX"
  python ./experiments/vit_incremental_cifar/vit_experiment.py --verbose --gpu_index 0 --run_index $INDEX \
  --config_file ./experiments/vit_incremental_cifar/config/local/single_runs/shrink_and_perturb.json
done
```
replace `config` with any config file in the `./experiments/vit_incremental_cifar/config/single_runs/` directory.
