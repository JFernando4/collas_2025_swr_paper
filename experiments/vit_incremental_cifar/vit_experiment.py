# built-in libraries
import time
import os
# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util import turn_off_debugging_processes, access_dict
# project files
from src import initialize_vit, initialize_vit_heads, initialize_layer_norm_module, initialize_multihead_self_attention_module, initialize_mlp_block
from src.utils import get_cifar_data, compute_accuracy_from_batch
from src.networks.torchvision_modified_vit import VisionTransformer
from src.networks import ReparameterizedLayerNorm, perturb_weights
from src import parse_terminal_arguments

from src.utils import save_model_parameters, set_random_seed
from src.swr_functions import get_network_init_parameters, SelectiveWeightReinitialization
from src.utils import IncrementalCIFARExperimentBase


class IncrementalCIFARExperiment(IncrementalCIFARExperimentBase):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True, gpu_index: int = 0):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        turn_off_debugging_processes(access_dict(exp_params, key="debug", default=True, val_type=bool))
        # define torch device
        gpu_index = access_dict(exp_params, "gpu_index", default=gpu_index, val_type=int)
        self.device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        set_random_seed(self.run_index)

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]

        # problem definition parameters
        self.num_epochs = access_dict(exp_params, "num_epochs", default=1, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=2, val_type=int)
        self.fixed_classes = access_dict(exp_params, "fixed_classes", default=True, val_type=bool)
        self.use_best_network = access_dict(exp_params, "use_best_network", default=True, val_type=bool)
        self.compare_loss = access_dict(exp_params, "compare_loss", default=False, val_type=bool)

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.rescaled_wd = access_dict(exp_params, "rescaled_wd", default=False, val_type=bool)
        self.wd_on_1d_params = access_dict(exp_params, "wd_on_1d_params", default=True, val_type=bool)
        self.momentum = exp_params["momentum"]
        self.reset_momentum = access_dict(exp_params, "reset_momentum", default=False, val_type=bool)
        self.use_lr_schedule = access_dict(exp_params, "use_lr_schedule", default=True, val_type=bool)
        self.dropout_prob = access_dict(exp_params, "dropout_prob", default=0.05, val_type=float)

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        self.reset_layer_norm = access_dict(exp_params, "reset_layer_norm", default=False, val_type=bool)
        self.reset_attention_layers = access_dict(exp_params, "reset_attention_layers", default=False, val_type=bool)
        self.reset_mlp_blocks = access_dict(exp_params, "reset_mlp_blocks", default=False, val_type=bool)

        # other network parameters
        self.reparam_ln = access_dict(exp_params, "reparam_ln", default=False, val_type=bool)

        # SWR parameters
        self.reinit_freq = access_dict(exp_params, "reinit_freq", default=0, val_type=int)
        self.reinit_factor = access_dict(exp_params, "reinit_factor", default=0.0, val_type=float)
        self.utility_function = access_dict(exp_params, "utility_function", default="none", val_type=str, choices=[ "none", "magnitude", "gradient"])
        self.pruning_method = access_dict(exp_params, "pruning_method", default="none", val_type=str, choices=["none", "proportional", "threshold"])
        self.reinit_strat = access_dict(exp_params, "reinit_strat", default="none", val_type=str, choices=["none", "mean", "resample"])
        self.use_swr = (self.pruning_method != "none") and (self.reinit_strat != "none") and (self.reinit_freq > 0) and (self.reinit_factor > 0.0)

        # CBP parameters
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=None, val_type=float)
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=None, val_type=int)
        self.use_cbp = (self.replacement_rate is not None) and (self.maturity_threshold is not None)

        # ReDO parameters
        self.redo_reinit_frequency = access_dict(exp_params, "redo_reinit_frequency", default=None, val_type=int)
        self.redo_reinit_threshold = access_dict(exp_params, "redo_reinit_threshold", default=None, val_type=float)
        self.use_redo = (self.redo_reinit_frequency is not None) and (self.redo_reinit_threshold is not None)

        # shrink and perturb parameters
        self.parameter_noise_var = access_dict(exp_params, "parameter_noise_var", default=0.0, val_type=float)
        self.use_parameter_noise = self.parameter_noise_var > 0.0

        """ Training constants """
        self.batch_sizes = {"train": 90, "test": 100, "validation":50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_epoch = 50000
        self.num_images_per_class = 450
        self.num_workers = 1 if self.device.type == "cpu" else 12   # for the data loader, change this to number of available cpus

        """ Network set up """
        # initialize network
        self.net = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=8,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=self.num_classes,
            dropout=self.dropout_prob,
            attention_dropout=self.dropout_prob,
            replacement_rate=self.replacement_rate,
            maturity_threshold=self.maturity_threshold,
            reinit_frequency=self.redo_reinit_frequency,
            reinit_threshold=self.redo_reinit_threshold,
            norm_layer=ReparameterizedLayerNorm if self.reparam_ln else torch.nn.LayerNorm
        )
        initialize_vit(self.net)
        self.net.to(self.device)

        # initialize optimizer and loss function
        self.optim = self._get_optimizer()
        self.lr_scheduler = None
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # initialize selective weight reinitialization
        self.swr_optim = None
        if self.use_swr:
            means, std, normal_reinit = get_network_init_parameters(self.net, self.reinit_strat, reparam_ln=self.reparam_ln)
            self.swr_optim = SelectiveWeightReinitialization(self.net.parameters(),
                                                             utility_function=self.utility_function,
                                                             pruning_method=self.pruning_method,
                                                             param_means=means,
                                                             param_stds=std,
                                                             normal_reinit=normal_reinit,
                                                             reinit_freq=self.reinit_freq,
                                                             reinit_factor=self.reinit_factor,
                                                             decay_rate=0.0)
        # initialize training counters
        self.current_epoch = 0
        self.current_minibatch = 0

        """ For data partitioning """
        self.class_increase = access_dict(exp_params, "class_increase", default=5, val_type=int)
        self.class_increase_frequency = 100
        self.all_classes = np.random.permutation(self.num_classes)  # define order classes
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_loss = torch.ones_like(self.best_accuracy) * torch.inf
        self.best_model_parameters = {}

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added
        self.delete_old_checkpoints = True

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

    # ------------------------------ Methods for initializing the experiment ------------------------------
    def _get_optimizer(self):
        """ Creates optimizer object based on the experiment parameters """
        if self.wd_on_1d_params:
            wd = self.weight_decay if self.rescaled_wd else self.weight_decay / self.stepsize
            params = self.net.parameters()
            return torch.optim.SGD(params, lr=self.stepsize, momentum=self.momentum, weight_decay=wd)
        else:
            # from Andrej Karpathy's nanoGPT repo: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py (lines 179-202)
            # start with all the candidate parameters (that require grad)
            param_dict = {pn: p for pn, p in self.net.named_parameters() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [{'params': decay_params, 'weight_decay': self.weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]
            return torch.optim.SGD(optim_groups, lr=self.stepsize, momentum=self.momentum)

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data
        training_data, training_dl = get_cifar_data(self.data_path, train=True, validation=False,
                                                    batch_size=self.batch_sizes["train"], num_workers=self.num_workers)
        val_data, val_dl = get_cifar_data(self.data_path, train=True, validation=True,
                                          batch_size=self.batch_sizes["validation"], num_workers=self.num_workers)
        test_data, test_dl = get_cifar_data(self.data_path, train=False, batch_size=self.batch_sizes["test"],
                                            num_workers=self.num_workers)
        # load checkpoint if available
        self.load_experiment_checkpoint()
        # train network
        self.train(train_dataloader=training_dl, test_dataloader=test_dl, val_dataloader=val_dl,
                   test_data=test_data, training_data=training_data, val_data=val_data)

        self.post_process_results()
        # if using mlproj_manager, summaries are stored in memory by calling exp.store_results()

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):

        # partition data
        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])

        # get lr scheduler and save model parameters
        if self.use_lr_schedule:
            self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))
        save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

        # start training
        for e in range(self.current_epoch, self.num_epochs):
            self._print(f"Epoch: {e + 1}")

            epoch_start = time.perf_counter()
            for step_number, sample in enumerate(tqdm(train_dataloader)):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # compute prediction and loss
                predictions = self.net.forward(image)[:, self.all_classes[:self.current_num_classes]]
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()

                # backpropagate and update weights
                current_loss.backward()
                self.optim.step()
                if self.use_lr_schedule:
                    self.lr_scheduler.step()
                    if self.lr_scheduler.get_last_lr()[0] > 0.0 and not self.rescaled_wd:
                        self.optim.param_groups[0]['weight_decay'] = self.weight_decay / self.lr_scheduler.get_last_lr()[0]
                # use swr
                if self.swr_optim is not None:
                    self.swr_optim.step()
                    self.store_num_replaced()
                # use shrink and perturb (shrink part is done by the optimizer)
                if self.use_parameter_noise: perturb_weights(self.net, self.parameter_noise_var)

                # store summaries
                current_accuracy = compute_accuracy_from_batch(predictions, label)
                self.running_loss += detached_loss
                self.running_accuracy += current_accuracy.detach()
                if (step_number + 1) % self.running_avg_window == 0:
                    # self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

                self.current_minibatch += 1

            epoch_end = time.perf_counter()

            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e, epoch_runtime=epoch_end - epoch_start)
            self.current_epoch += 1

            self.extend_classes(training_data, test_data, val_data, train_dataloader)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def get_lr_scheduler(self, steps_per_epoch: int):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.stepsize, anneal_strategy="linear",
                                                        epochs=self.class_increase_frequency,
                                                        steps_per_epoch=steps_per_epoch)
        if not self.rescaled_wd:
            self.optim.param_groups[0]['weight_decay'] = self.weight_decay / scheduler.get_last_lr()[0]
        return scheduler

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet,
                       train_dataloader: DataLoader):
        """
        Adds 5 new classes to the data set with certain frequency
        """
        if (self.current_epoch % self.class_increase_frequency) == 0 and (not self.fixed_classes):
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))
            if self.use_best_network:
                self.net.load_state_dict(self.best_model_parameters)
            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_loss = torch.ones_like(self.best_accuracy) * torch.inf
            self.best_model_parameters = {}
            save_model_parameters(self.results_dir, self.run_index, self.current_epoch, self.net)

            if self.current_num_classes == self.num_classes: return

            self.current_num_classes += self.class_increase
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew class added...")
            if self.reset_head:
                initialize_vit_heads(self.net.heads)
            if self.reset_network:
                initialize_vit(self.net)
                self.optim = self._get_optimizer()
            if self.reset_layer_norm:
                self.net.apply(initialize_layer_norm_module)
            if self.reset_attention_layers:
                self.net.apply(initialize_multihead_self_attention_module)
            if self.reset_mlp_blocks:
                self.net.apply(initialize_mlp_block)
            if self.reset_momentum:
                self.optim = self._get_optimizer()
            if self.use_lr_schedule:
                self.lr_scheduler = self.get_lr_scheduler(steps_per_epoch=len(train_dataloader))
            return True
        return False


def main():
    """
    Function for running the experiment from command line given a path to a json config file
    """
    from mlproj_manager.file_management.file_and_directory_management import read_json_file
    terminal_arguments = parse_terminal_arguments()
    experiment_parameters = read_json_file(terminal_arguments.config_file)
    file_path = os.path.dirname(os.path.abspath(__file__))

    experiment_parameters["data_path"] = os.path.join(file_path, "data")
    print(experiment_parameters)
    relevant_parameters = experiment_parameters["relevant_parameters"]
    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    results_path = os.path.join(file_path, "results") if "results_path" not in experiment_parameters.keys() else experiment_parameters["results_path"]

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(results_path, results_dir_name),
                                     run_index=terminal_arguments.run_index,
                                     verbose=terminal_arguments.verbose,
                                     gpu_index=terminal_arguments.gpu_index)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
