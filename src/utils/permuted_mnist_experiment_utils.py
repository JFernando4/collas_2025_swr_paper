# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
# ml project manager
from mlproj_manager.experiments import Experiment
# from src
from src.networks import ThreeHiddenLayerNetwork
from src.utils.evaluation_functions import compute_matrix_rank_summaries, compute_average_weight_magnitude, compute_average_gradient_magnitude


@torch.no_grad()
def compute_dead_units_prop_and_stable_rank(net: ThreeHiddenLayerNetwork, data_loader: DataLoader, num_activations: int,
                                            batch_size: int = 30, num_mini_batches: int = 50,
                                            activation_function: str = "relu", epsilon: float = 0.01):
    """
    Computes the proportion of dead units and the stable rank of the representation (the last layer)
    """
    num_inputs = 784
    num_layers = 3

    # compute some number of activations
    all_activations = torch.zeros((num_mini_batches * batch_size, num_layers, num_activations), dtype=torch.float32)
    for i, sample in enumerate(data_loader):
        if i >= num_mini_batches:
            break
        image = sample["image"].reshape(batch_size, num_inputs)
        temp_acts = []
        net.forward(image, activations=temp_acts)
        for l in range(num_layers):
            all_activations[i * batch_size:(i + 1) * batch_size, l, :] = temp_acts[l]

    if activation_function in ["relu", "leaky_relu"]:
        prop_dead_units = torch.mean((all_activations.sum(0) == 0.0).to(torch.float32)).item()
    elif activation_function in ["sigmoid", "tanh", "gelu", "silu"]:
        if activation_function == "sigmoid":
            bounds = (0, 1)
        elif activation_function == "tanh":
            bounds = (-1, 1)
        elif activation_function == "gelu" or activation_function == "silu":
            bounds = (0, np.inf)
        else:
            raise ValueError(f"Unknown activation function '{activation_function}'")
        # check if the mean activation is below the lower bound or above the upper bound
        lower_than_lower_bound = all_activations <= bounds[0] + epsilon
        higher_than_upper_bound = all_activations >= bounds[1] + epsilon
        frozen_below = torch.all(lower_than_lower_bound, dim=0)
        frozen_above = torch.all(higher_than_upper_bound, dim=0)
        # mean_across_batches = all_activations.mean(0).flatten()
        # dormant_below = mean_across_batches <= bounds[0] + epsilon
        # dormant_above = mean_across_batches >= bounds[1] - epsilon
        # technically, this is the proportion of units that are frozen
        prop_dead_units = torch.logical_or(frozen_below, frozen_above).to(torch.float32).mean().item()
        # print(f"Proportion of dormant units: {prop_dead_units:.4f}\tdormant below: {dormant_below.to(torch.float32).mean().item()}\tdormant above: {dormant_above.to(torch.float32).mean().item()}")
    else:
        raise ValueError(f"Unknown activation function '{activation_function}'")

    _, _, _, stable_rank = compute_matrix_rank_summaries(all_activations[:, -1, :], prop=0.99, use_scipy=False,
                                                         return_rank=False, return_effective_rank=False,
                                                         return_approximate_rank=False)
    return prop_dead_units, stable_rank.item()


@torch.no_grad()
def compute_dead_units_proportion(net: ThreeHiddenLayerNetwork, data_loader: DataLoader, num_activations: int = 10,
                                  batch_size: int = 30, num_inputs: int = 784,  num_mini_batches: int = 50):
    """ computes the proportion of dead units in the network"""

    num_layers = 3
    all_activations = torch.zeros((num_mini_batches * batch_size, num_layers * num_activations), dtype=torch.float32)
    for i, sample in enumerate(data_loader):

        if i >= num_mini_batches:
            break

        image = sample["image"].reshape(batch_size, num_inputs)
        temp_acts = []
        net.forward(image, activations=temp_acts)

        stacked_act = torch.hstack(temp_acts)
        all_activations[i * batch_size:(i+1) * batch_size, :] = stacked_act

    sum_activations = all_activations.sum(0)
    return torch.mean((sum_activations == 0.0).to(torch.float32))


class PermutedMNISTExperimentBase(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=False):
        super().__init__(exp_params, results_dir, run_index, verbose)

        self.device = torch.device("cpu")
        self.compute_average_weight_magnitude = 0
        self.extended_summaries = False
        self.num_permutations = 0
        self.steps_per_task = 0
        self.current_experiment_step = 0

        self.swr_optim = None
        self.reinit_freq = 0
        self.num_hidden = 0
        self.use_swr = False
        self.use_cbp = False
        self.use_redo = False
        self.use_ln = False
        self.activation_function = ""

        self.num_classes = 10
        self.num_inputs = 784

        """ Network set up """
        self.net = None

        """ Experiment Summaries """
        self.batch_size = 30
        self.running_avg_window = 10
        self.store_next_loss = False        # indicates whether to store the loss computed on the next batch
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.current_permutation = (0, 0.0, 0.0, 0)
        self.running_avg_grad_magnitude = 0.0
        self.previous_activations = []
        self.results_dict = {}

    # ----------------------------- For storing summaries ----------------------------- #
    def initialize_results_dict(self) -> dict:
        """
        Initializes the results dictionary for the permuted mnist experiment
        """
        results_dict = {}
        defaults = {"device": self.device, "dtype": torch.float32}

        total_ckpts = self.steps_per_task * self.num_permutations // (self.running_avg_window * self.batch_size)
        results_dict["train_loss_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)
        results_dict["train_accuracy_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)

        if self.use_swr or self.use_redo or self.use_cbp:
            if self.extended_summaries:
                results_dict["num_replaced"] = []
                results_dict["loss_before_topology_update"] = []
                results_dict["loss_after_topology_update"] = []
                results_dict["avg_grad_before_topology_update"] = []
                results_dict["avg_grad_after_topology_update"] = []
                if self.use_ln:
                    results_dict["change_in_average_activation_layer_1"] = []
                    results_dict["change_in_average_activation_layer_2"] = []
                    results_dict["change_in_average_activation_layer_3"] = []
                    results_dict["change_in_std_activation_layer_1"] = []
                    results_dict["change_in_std_activation_layer_2"] = []
                    results_dict["change_in_std_activation_layer_3"] = []

        if self.extended_summaries:
            results_dict["average_gradient_magnitude_per_checkpoint"] = torch.zeros(total_ckpts, **defaults)
            results_dict["average_weight_magnitude_per_permutation"] = torch.zeros(self.num_permutations, **defaults)
            results_dict["proportion_dead_units_per_permutation"] = torch.zeros(self.num_permutations, **defaults)
            results_dict["stable_rank_per_permutation"] = torch.zeros(self.num_permutations, **defaults)
            if self.use_ln:
                results_dict["average_ln_weight_magnitude_per_checkpoint"] = torch.zeros(self.num_permutations, **defaults)

        return results_dict

    def _store_training_summaries(self):
        # store train data for checkpoints
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0

        if self.extended_summaries:
            self.results_dict["average_gradient_magnitude_per_checkpoint"][self.current_running_avg_step] += \
                self.running_avg_grad_magnitude / self.running_avg_window
            self.running_avg_grad_magnitude *= 0.0

        self.current_running_avg_step += 1

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def compute_network_extended_summaries(self, training_data: DataLoader):
        """ Computes the average weight magnitude, proportion of dead units, and stable rank of the representation """
        if not self.extended_summaries: return
        avg_weight_magnitude, avg_ln_weight_magnitude = compute_average_weight_magnitude(self.net)
        prop_dead_units, stable_rank = compute_dead_units_prop_and_stable_rank(self.net, training_data, self.num_hidden,
                                                                               self.batch_size,
                                                                               activation_function=self.activation_function,
                                                                               epsilon=0.01)
        self.results_dict["average_weight_magnitude_per_permutation"][self.current_permutation] += avg_weight_magnitude
        self.results_dict["stable_rank_per_permutation"][self.current_permutation] += stable_rank
        self.results_dict["proportion_dead_units_per_permutation"][self.current_permutation] += prop_dead_units
        if self.use_ln:
            self.results_dict["average_ln_weight_magnitude_per_checkpoint"][self.current_permutation] += avg_ln_weight_magnitude

    def store_extended_summaries(self, current_loss: torch.Tensor, current_activations: list = None) -> None:
        """ Stores the extended summaries related to the topology update of CBP and CBPw """
        if not self.extended_summaries: return

        store_cbp_summaries = self.use_cbp and self.net.feature_replace_event_indicator()
        store_redo_summaries = self.use_redo and self.net.feature_replace_event_indicator()
        store_swr_summaries = self.use_swr and self.swr_optim.reinit_indicator

        if (not store_cbp_summaries and         # check if using cbp and a feature has been replaced
            not store_swr_summaries and         # check if using swr and weights have been replaced
            not store_redo_summaries and        # check if using redo
            not self.store_next_loss):          # check if cbp, swr, or redo was used in the previous step
            return

        # reinitialization happened on the current step
        if not self.store_next_loss and (store_cbp_summaries or store_swr_summaries or store_redo_summaries):
            self.store_before_reinitialization_summaries(current_loss)
            if self.use_ln:
                self.previous_activations = current_activations
            self.store_num_replace_summary()

        # reinitialization happened on the previous step
        elif self.store_next_loss and (not store_cbp_summaries and not store_swr_summaries and not store_redo_summaries):
            self.store_after_reinitialization_summaries(current_loss)
            if self.use_ln:
                self.store_change_in_activation_statistics_summaries(current_activations)
                self.previous_activations = []

        # reinitialization happened on the current and previous step
        elif self.store_next_loss and (store_swr_summaries or store_swr_summaries or store_redo_summaries):
            self.store_before_reinitialization_summaries(current_loss)
            self.store_after_reinitialization_summaries(current_loss)
            if self.use_ln:
                self.store_change_in_activation_statistics_summaries(current_activations)
                self.previous_activations = current_activations
            self.store_num_replace_summary()

        self.store_next_loss = store_cbp_summaries or store_swr_summaries or store_redo_summaries
        if self.swr_optim is not None:
            self.swr_optim.reset_reinit_indicator()
        self.net.reset_indicators()

    def store_before_reinitialization_summaries(self, current_loss: torch.Tensor):
        self.results_dict["loss_before_topology_update"].append(current_loss)
        self.results_dict["avg_grad_before_topology_update"].append(compute_average_gradient_magnitude(self.net))

    def store_after_reinitialization_summaries(self, current_loss: torch.Tensor):
        self.results_dict["loss_after_topology_update"].append(current_loss)
        self.results_dict["avg_grad_after_topology_update"].append(compute_average_gradient_magnitude(self.net))

    def store_change_in_activation_statistics_summaries(self, current_activations: list[torch.Tensor]):
        for i in range(len(current_activations)):
            diff_average_act = current_activations[i].mean().detach() - self.previous_activations[i].mean().detach()
            diff_std_act = current_activations[i].std().detach() - self.previous_activations[i].std().detach()
            self.results_dict[f"change_in_average_activation_layer_{i + 1}"].append(diff_average_act.abs())
            self.results_dict[f"change_in_std_activation_layer_{i + 1}"].append(diff_std_act.abs())

    def store_num_replace_summary(self):
        if not self.use_cbp and not self.use_redo and not self.use_swr: return
        num_replaced = self.swr_optim.num_replaced if self.use_swr else sum(self.net.num_replaced())
        self.results_dict["num_replaced"].append(num_replaced)

    def post_process_extended_results(self):
        using_cbp_or_swr_or_redo = self.use_cbp or self.use_swr or self.use_redo
        if not self.extended_summaries or not using_cbp_or_swr_or_redo: return
        for k in self.results_dict.keys():
            if not isinstance(self.results_dict[k], np.ndarray):
                self.results_dict[k] = np.array(self.results_dict[k], dtype=np.float32)
