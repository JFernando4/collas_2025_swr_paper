# built-in libraries
import time
import os

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# from ml project manager
from mlproj_manager.problems import MnistDataSet
from mlproj_manager.util import access_dict, Permute, turn_off_debugging_processes
from mlproj_manager.util.neural_networks import init_weights_kaiming

# from src
from src.networks import ThreeHiddenLayerNetwork, init_three_hidden_layer_network_weights, perturb_weights
from src.utils.experiment_utils import parse_terminal_arguments
from src.utils.evaluation_functions import compute_average_gradient_magnitude, set_random_seed
from src.utils.permuted_mnist_experiment_utils import PermutedMNISTExperimentBase
from src.swr_functions import SelectiveWeightReinitialization, get_network_init_parameters
from src.optimizers import SGDW


class PermutedMNISTExperiment(PermutedMNISTExperimentBase):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=False):
        super().__init__(exp_params, results_dir, run_index, verbose=verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cpu")

        """ For reproducibility """
        set_random_seed(self.run_index)

        """ Experiment parameters """
        self.extended_summaries = access_dict(exp_params, "extended_summaries", default=False, val_type=bool)
        # learning parameters
        self.stepsize = exp_params["stepsize"]
        self.l2_factor = access_dict(exp_params, "l2_factor", default=0.0, val_type=float)
        self.use_adamw = access_dict(exp_params, "use_adamw", default=False, val_type=bool)
        self.momentum = access_dict(exp_params, "momentum", default=0.0, val_type=float)
        self.beta2 = access_dict(exp_params, "beta2", default=None, val_type=float)

        """ Architecture parameters """
        self.num_hidden = exp_params["num_hidden"]      # number of hidden units per hidden layer
        self.activation_function = access_dict(exp_params, "activation_function", default="relu", val_type=str,
                                               choices=["relu", "sigmoid", "tanh", "leaky_relu", "gelu", "silu"])
        self.use_crelu = access_dict(exp_params, "use_crelu", default=False, val_type=bool)
        if self.use_crelu and self.activation_function != "relu":
            raise ValueError("The use_crelu parameter is only valid when the activation function is relu.")
        # layer norm parameters
        self.use_ln = access_dict(exp_params, "use_ln", default=False, val_type=bool)
        self.preactivation_ln = access_dict(exp_params, "preactivation_ln", default=False, val_type=bool)
        # residual network parameters
        self.use_skip_connections = access_dict(exp_params, "use_skip_connections", default=False, val_type=bool)
        self.preactivation_skip_connections = access_dict(exp_params, "preactivation_skip_connections", default=False, val_type=bool)

        # problem parameters
        self.num_permutations = exp_params["num_permutations"]      # 1 permutation = 1 epoch
        self.batch_size = 30
        self.steps_per_task = 60000
        self.current_experiment_step = 0

        """ Reinitialization parameters """
        # SWR parameters
        self.reinit_freq = access_dict(exp_params, "reinit_freq", default=0, val_type=int)
        self.reinit_factor = access_dict(exp_params, "reinit_factor", default=0.0, val_type=float)
        self.utility_function = access_dict(exp_params, "utility_function", default="none", val_type=str, choices=["none", "magnitude", "gradient"])
        self.pruning_method = access_dict(exp_params, "pruning_method", default="none", val_type=str, choices=["none", "proportional", "threshold"])
        self.reinit_strat = access_dict(exp_params, "reinit_strat", default="none", val_type=str, choices=["none", "resample", "mean", "random"])
        self.use_swr = (self.pruning_method != "none") and (self.reinit_strat != "none") and (self.reinit_freq > 0) and (self.reinit_factor > 0.0)
        # cbp parameters
        feature_utility_names = ["none", "contribution", "magnitude", "gradient", "activation"]
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=None, val_type=int)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=None, val_type=float)
        self.cbp_utility = access_dict(exp_params, "cbp_utility", default=None, val_type=str, choices=feature_utility_names)
        self.use_cbp = (self.maturity_threshold is not None) and (self.replacement_rate is not None) and (self.cbp_utility is not None)
        # redo parameters
        self.redo_reinit_freq = access_dict(exp_params, "redo_reinit_freq", default=None, val_type=int)
        self.redo_reinit_threshold = access_dict(exp_params, "redo_reinit_threshold", default=None, val_type=float)
        self.redo_utility = access_dict(exp_params, "redo_utility", default=None, val_type=str, choices=feature_utility_names)
        self.use_redo = (self.redo_reinit_freq is not None) and (self.redo_reinit_threshold is not None) and (self.redo_utility is not None)
        # for both cbp and redo
        self.reinit_after_ln = access_dict(exp_params, "reinit_after_ln", default=False, val_type=bool)
        # Shrink and Perturb parameters
        self.parameter_noise_var = access_dict(exp_params, "parameter_noise_var", default=0.0, val_type=float)
        self.use_parameter_noise = self.parameter_noise_var > 0.0
        # paths for loading and storing data
        self.use_reinit = self.use_swr or self.use_cbp or self.use_redo
        self.data_path = exp_params["data_path"]
        self.results_dir = results_dir

        """ Training constants """
        self.num_classes = 10
        self.num_inputs = 784

        """ Network set up """
        self.net = ThreeHiddenLayerNetwork(hidden_dim=self.num_hidden,
                                           activation_function=self.activation_function,
                                           use_skip_connections=self.use_skip_connections,
                                           preactivation_skip_connection=self.preactivation_skip_connections,
                                           use_cbp=self.use_cbp,
                                           maturity_threshold=self.maturity_threshold,
                                           replacement_rate=self.replacement_rate,
                                           cbp_utility=self.cbp_utility,
                                           use_redo=self.use_redo,
                                           reinit_frequency=self.redo_reinit_freq,
                                           reinit_threshold=self.redo_reinit_threshold,
                                           redo_utility=self.redo_utility,
                                           use_layer_norm=self.use_ln,
                                           preactivation_layer_norm=self.preactivation_ln,
                                           reinit_after_ln=self.reinit_after_ln,
                                           use_crelu=self.use_crelu)
        self.net.apply(lambda z: init_three_hidden_layer_network_weights(z, nonlinearity=self.activation_function))     # initialize weights
        self.net.to(self.device)
        # initialize selective weight reinitialization
        self.swr_optim = None
        if self.use_swr:
            means, std, normal_reinit = get_network_init_parameters(self.net, self.reinit_strat, reparam_ln=False)
            self.swr_optim = SelectiveWeightReinitialization(self.net.parameters(),
                                                             utility_function=self.utility_function,
                                                             pruning_method=self.pruning_method,
                                                             param_means=means,
                                                             param_stds=std,
                                                             normal_reinit=normal_reinit,
                                                             reinit_freq=self.reinit_freq,
                                                             reinit_factor=self.reinit_factor,
                                                             decay_rate=0.0)
        # initialize optimizer
        self.optim = self.get_optimizer()
        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        """ Experiment Summaries """
        self.running_avg_window = 10
        self.store_next_loss = False        # indicates whether to store the loss computed on the next batch
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.current_permutation = (0, 0.0, 0.0, 0)
        self.running_avg_grad_magnitude = 0.0
        self.previous_activations = []
        self.results_dict = self.initialize_results_dict()

    def get_optimizer(self):
        if self.use_adamw:
            betas = (self.momentum, self.momentum) if self.beta2 is None else (self.momentum, self.beta2)
            return torch.optim.AdamW(self.net.parameters(), lr=self.stepsize, weight_decay=self.l2_factor/self.stepsize, betas=betas)
        else:
            return SGDW(self.net.parameters(), lr=self.stepsize, weight_decay=self.l2_factor/self.stepsize, momentum=self.momentum)

    # --------------------------- For running the experiment --------------------------- #
    def run(self):
        # load data
        mnist_train_data = MnistDataSet(root_dir=self.data_path, train=True, device=self.device,
                                        image_normalization="max", label_preprocessing="one-hot", use_torch=True)
        mnist_data_loader = DataLoader(mnist_train_data, batch_size=self.batch_size, shuffle=True)

        # train network
        self.train(mnist_data_loader=mnist_data_loader, training_data=mnist_train_data)
        self.post_process_extended_results()
        print(np.average(self.results_dict["train_accuracy_per_checkpoint"]))

    def train(self, mnist_data_loader: DataLoader, training_data: MnistDataSet):

        for _ in tqdm(range(self.num_permutations), disable=not self.verbose):
            if self.current_permutation == self.num_permutations: break
            training_data.set_transformation(Permute(np.random.permutation(self.num_inputs)))  # apply new permutation

            # compute percent of dead units, stable rank, and average weight magnitude
            self.compute_network_extended_summaries(mnist_data_loader)

            # train for one task
            for i, sample in enumerate(mnist_data_loader):
                self.current_experiment_step += 1

                # sample observation and target
                image = sample["image"].reshape(self.batch_size, self.num_inputs)
                label = sample["label"]

                # reset gradients
                for param in self.net.parameters(): param.grad = None  # apparently faster than optim.zero_grad()

                # compute prediction and loss
                current_activations = [] if (self.extended_summaries and self.use_ln and (self.use_cbp or self.use_swr or self.use_redo)) else None
                predictions = self.net.forward(image, current_activations)
                current_loss = self.loss(predictions, label)
                detached_loss = current_loss.detach().clone()

                # backpropagate, update weights, use swr, and perturb weights
                current_loss.backward()
                self.optim.step()
                if self.swr_optim is not None: self.swr_optim.step()
                if self.use_parameter_noise: perturb_weights(self.net, self.parameter_noise_var)

                # store extended summaries
                if self.extended_summaries:
                    self.running_avg_grad_magnitude += compute_average_gradient_magnitude(self.net)
                self.store_extended_summaries(detached_loss, current_activations)

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += detached_loss
                self.running_accuracy += current_accuracy.detach()
                if (i + 1) % self.running_avg_window == 0:
                    self._store_training_summaries()

            self.current_permutation += 1


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    from mlproj_manager.file_management.file_and_directory_management import read_json_file
    terminal_arguments = parse_terminal_arguments()
    experiment_parameters = read_json_file(terminal_arguments.config_file)
    file_path = os.path.dirname(os.path.abspath(__file__))

    experiment_parameters["data_path"] = os.path.join(file_path, "data")
    relevant_parameters = experiment_parameters["relevant_parameters"]

    results_dir_name = "{0}-{1}".format(relevant_parameters[0], experiment_parameters[relevant_parameters[0]])
    for relevant_param in relevant_parameters[1:]:
        results_dir_name += "_" + relevant_param + "-" + str(experiment_parameters[relevant_param])

    results_path = os.path.join(file_path, "results") if "results_path" not in experiment_parameters.keys() else experiment_parameters["results_path"]

    initial_time = time.perf_counter()
    exp = PermutedMNISTExperiment(experiment_parameters,
                                  results_dir=os.path.join(results_path, results_dir_name),
                                  run_index=terminal_arguments.run_index,
                                  verbose=terminal_arguments.verbose)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
