# built-in libraries
import time
import os
import pickle
from copy import deepcopy
# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
# from ml project manager
from mlproj_manager.experiments import Experiment

from src.utils import evaluate_network


class IncrementalCIFARExperimentBase(Experiment):
    """ Base class with all the utilities for storing results and checkpoints """


    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

    # ------------------------------ Methods for initializing the experiment ------------------------------
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        if self.fixed_classes:
            num_images_per_epoch = self.num_images_per_class * self.num_classes
            total_checkpoints = (num_images_per_epoch * self.num_epochs) // (
                        self.running_avg_window * self.batch_sizes["train"])
        else:
            number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
            number_of_image_per_task = self.num_images_per_class * self.class_increase
            bin_size = (self.running_avg_window * self.batch_sizes["train"])
            total_checkpoints = np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size)

        train_prototype_array = torch.zeros(total_checkpoints, device=self.device, dtype=torch.float32)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros_like(train_prototype_array)

        prototype_array = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros_like(prototype_array)
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = torch.zeros_like(prototype_array)
        self.results_dict["class_order"] = self.all_classes

        # swr masks summaries
        if self.use_swr:
            self.results_dict["num_reinit"] = []

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "swr_optim_state": self.swr_optim.state_dict() if self.use_swr else None,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch_number": self.current_epoch,
            "minibatch_number": self.current_minibatch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results
        }

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were succesfully loaded
        """

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        self.net.load_state_dict(checkpoint["model_weights"])
        self.optim.load_state_dict(checkpoint["optim_state"])
        if self.use_swr:
            self.swr_optim.load_state_dict(checkpoint["swr_optim_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        self.current_epoch = checkpoint["epoch_number"]
        self.current_minibatch = checkpoint["minibatch_number"]
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            if k not in partial_results.keys():  # delete this line and the one below
                continue
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else \
            partial_results[k].to(self.device)


    # --------------------------------------- For storing summaries --------------------------------------- #
    def _store_training_summaries(self):
        self.results_dict["train_loss_per_checkpoint"][
            self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][
            self.current_running_avg_step] += self.running_accuracy / self.running_avg_window

        # self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int,
                              epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            # evaluate on data
            evaluation_start_time = time.perf_counter()
            loss, accuracy = evaluate_network(data_loader, self.device, self.loss, self.net, self.all_classes,
                                              self.current_num_classes)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    if not self.compare_loss:
                        self.best_model_parameters = deepcopy(self.net.state_dict())
                if loss < self.best_loss:
                    self.best_loss = loss
                    if self.compare_loss:
                        self.best_model_parameters = deepcopy(self.net.state_dict())

            # store summaries
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time,
                                                                                               dtype=torch.float32)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy

            # print progress
            self._print(f"\t{data_name} accuracy: {accuracy:.4f}")

        self.net.train()
        # self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def store_num_replaced(self):
        if not self.use_swr: return

        if self.swr_optim.reinit_indicator:
            self.results_dict["num_reinit"].append(self.swr_optim.num_replaced)
            self.swr_optim.reset_reinit_indicator()

    def post_process_results(self):
        if self.use_swr:
            self.results_dict["num_reinit"] = np.array(self.results_dict["num_reinit"])
