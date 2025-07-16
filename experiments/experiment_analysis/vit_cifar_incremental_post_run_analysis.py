# built-in libraries
import time
import os
import argparse

# third party libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from scipy.linalg import svd

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize
from mlproj_manager.file_management import read_json_file
from mlproj_manager.util.experiments_util import access_dict

# from source
from src.networks.torchvision_modified_vit import VisionTransformer
from src.networks import ReparameterizedLayerNorm
from src.utils import parse_plots_and_analysis_terminal_arguments

DEBUG = False

# -------------------- For loading data and network parameters -------------------- #
def load_model_parameters(parameter_dir_path: str, index: int, epoch_number:int):
    """
    Loads the model parameters stored in parameter_dir_path corresponding to the index and epoch number
    return: torch module state dictionary
    """

    model_parameters_file_name = "index-{0}_epoch-{1}.pt".format(index, epoch_number)
    model_parameters_file_path = os.path.join(parameter_dir_path, model_parameters_file_name)

    if not os.path.isfile(model_parameters_file_path):
        error_message = "Couldn't find model parameters for index {0} and epoch number {1}.".format(index, epoch_number)
        raise ValueError(error_message)

    return torch.load(model_parameters_file_path)


def load_classes(classes_dir_path: str, index: int):
    """
    Loads the list of ordered classes used for partitioning the datta during the experiment
    return: list
    """

    classes_file_name = "index-{0}.npy".format(index)
    classes_file_path = os.path.join(classes_dir_path, classes_file_name)

    if not os.path.isfile(classes_file_path):
        error_message = "Couldn't find list of classes for index {0}.".format(index)
        raise ValueError(error_message)

    return np.load(classes_file_path)


def load_cifar_data(data_path: str, train: bool = True) -> (CifarDataSet, DataLoader):
    """
    Loads the cifar 100 data set with normalization
    :param data_path: path to the directory containing the data set
    :param train: bool that indicates whether to load the train or test data
    :return: torch DataLoader object
    """
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=train,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]

    cifar_data.set_transformation(transforms.Compose(transformations))

    num_workers = 12
    batch_size = 1000
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return cifar_data, dataloader


# -------------------- For computing analysis of the network -------------------- #
@torch.no_grad()
def compute_dormant_units_proportion(net: VisionTransformer, cifar_data_loader: DataLoader, epsilon: float = 0.01):
    """
    Computes the proportion of dormant units in a VisionTrandformer. It also returns the features of the last layer for the first
    1000 samples
    """

    device = net.heads[0].weight.device
    features_per_layer = []
    last_layer_activations = None
    num_samples = 1000

    for i, sample in enumerate(cifar_data_loader):
        image = sample["image"].to(device)
        temp_features = []
        net.forward(image, temp_features)

        features_per_layer = temp_features
        last_layer_activations = temp_features[-1][:, 0,:].cpu()
        break

    frozen_units = torch.zeros(len(features_per_layer), dtype=torch.float32)
    number_of_features = 0
    for layer_idx in range(len(features_per_layer)):
        frozen_units[layer_idx] = torch.all(features_per_layer[layer_idx] <= epsilon, dim=0).to(torch.float32).sum()
        number_of_features += features_per_layer[layer_idx].shape[1]
    if DEBUG: print(f"Shape of the last layer activations {last_layer_activations.shape}")
    return frozen_units.sum().item() / number_of_features, last_layer_activations.numpy()


def compute_stable_rank(singular_values: np.ndarray):
    """ Computes the stable rank of the representation layer """
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


@torch.no_grad()
def compute_last_task_accuracy_per_class_in_order(net: torch.nn.Module, ordered_classes: np.ndarray,
                                                  test_data: DataLoader, experiment_index: int):
    """
    Computes the accuracy of each class in the order they were presented
    :param net: resnet with the parameters stored at the end of the experiment
    :param ordered_classes: numpy array with the cifar 100 classes in the order they were presented
    :param test_data: cifar100 test data
    :return: numpy array
    """

    ordered_classes = np.int32(ordered_classes)
    device = net.heads[0].weight.device
    num_classes = 100
    num_examples_per_class = 100

    class_correct = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for i, sample in enumerate(test_data):
        image = sample["image"].to(device)
        labels = sample["label"].to(device)
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)    # Get the class with the highest score
        _, labels = torch.max(labels, 1)        # Get the class with the highest score

        # Update the counts for each class
        for i, class_label in enumerate(ordered_classes):
            class_correct[i] += (predicted == labels).masked_select(labels == class_label).sum().item()

    return class_correct.cpu().numpy() / num_examples_per_class


# -------------------- For storing the results of the analysis -------------------- #
def store_analysis_results(dormant_units_results: (np.ndarray, np.ndarray),
                           stable_rank_results: (np.ndarray, np.ndarray),
                           accuracy_per_class_in_order: np.ndarray,
                           results_dir: str, experiment_index: int):
    """
    Stores the results of the post run analysis
    :param dormant_units_results: tuple containing the results of the dormant unit analysis for the previous tasks and
                                  the next task for each different task
    :param stable_rank_results: tuple containing the results of the stable rank analysis for the previous tasks and the
                                next task for each different task
    :param accuracy_per_class_in_order: np array containing the accuracy of the final model for each class in the order
                                        they were presented
    :param results_dir: path to the results directory
    :param experiment_index: experiment index
    """

    index_file_name = "index-{0}.npy".format(experiment_index)
    result_dir_names_and_arrays = [
        ("previous_tasks_dormant_units_analysis", dormant_units_results[0]),
        ("next_task_dormant_units_analysis", dormant_units_results[1]),
        ("previous_tasks_stable_rank_analysis", stable_rank_results[0]),
        ("next_task_stable_rank_analysis", stable_rank_results[1]),
        ("accuracy_per_class_in_order", accuracy_per_class_in_order)
    ]

    # store results in the corresponding dir
    for results_name, results_array in result_dir_names_and_arrays:
        temp_results_dir = os.path.join(results_dir, results_name)
        os.makedirs(temp_results_dir, exist_ok=True)
        np.save(os.path.join(temp_results_dir, index_file_name), results_array)


def analyze_results(analysis_parameters: dict):
    """
    Analyses the parameters of a run and creates files with the results of the analysis
    """

    results_dir = analysis_parameters["results_dir"]
    data_path = analysis_parameters["data_path"]
    net_parameters = analysis_parameters["net_parameters"]
    dormant_unit_threshold = access_dict(analysis_parameters, "dormant_units_results", default=0.01, val_type=float)
    excluded_indices = access_dict(analysis_parameters, "excluded_indices", default=[], val_type=list)

    parameter_dir_path = os.path.join(results_dir, "model_parameters")
    experiment_indices_file_path = os.path.join(results_dir, "experiment_indices.npy")
    class_order_dir_path = os.path.join(results_dir, "class_order")

    gpu_index = 0 if "gpu_index" not in net_parameters else net_parameters["gpu_index"]
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    number_of_epochs = np.arange(21) * 100  # by design the model parameters where store after each of these epochs
    classes_per_task = 5                    # by design each task increases the data set by 5 classes
    last_epoch = 2000
    experiment_indices = np.load(experiment_indices_file_path)

    net = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=8,
            num_heads=12,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=100,
            dropout=net_parameters["dropout"],
            attention_dropout=net_parameters["dropout"],
            norm_layer=ReparameterizedLayerNorm if net_parameters["reparam_ln"] else torch.nn.LayerNorm,
            replacement_rate=None if "replacement_rate" not in net_parameters else net_parameters["replacement_rate"],
            maturity_threshold=None if "maturity_threshold" not in net_parameters else net_parameters["maturity_threshold"],
            reinit_frequency=None if "reinit_frequency" not in net_parameters else net_parameters["reinit_frequency"],
            reinit_threshold=None if "reinit_threshold" not in net_parameters else net_parameters["reinit_threshold"],

    )
    net.to(device)
    net.eval()
    cifar_data, cifar_data_loader = load_cifar_data(data_path, train=True)
    test_data, test_data_loader = load_cifar_data(data_path, train=False)

    for exp_index in experiment_indices:
        if exp_index in excluded_indices:
            continue

        print("Experiment index: {0}".format(exp_index))
        ordered_classes = load_classes(class_order_dir_path, index=exp_index)

        dormant_units_prop_before = np.zeros(number_of_epochs.size - 1, dtype=np.float32)
        stable_rank_before = np.zeros_like(dormant_units_prop_before)
        dormant_units_prop_after = np.zeros_like(dormant_units_prop_before)
        stable_rank_after = np.zeros_like(dormant_units_prop_before)

        for i, epoch_number in enumerate(number_of_epochs[:-1]):
            # get model parameters from before training on the task
            model_parameters = load_model_parameters(parameter_dir_path, index=exp_index, epoch_number=epoch_number)
            net.load_state_dict(model_parameters)

            # compute summaries for next task
            current_classes = ordered_classes[(i * classes_per_task):((i + 1) * classes_per_task)]
            cifar_data.select_new_partition(current_classes)

            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)
            dormant_units_prop_after[i] = prop_dormant
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            stable_rank_after[i] = compute_stable_rank(singular_values)

            # compute summaries from data from previous tasks
            if i == 0: continue
            current_classes = ordered_classes[:(i * classes_per_task)]
            cifar_data.select_new_partition(current_classes)
            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)

            dormant_units_prop_before[i] = prop_dormant
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            stable_rank_before[i] = compute_stable_rank(singular_values)

        net.load_state_dict(load_model_parameters(parameter_dir_path, exp_index, last_epoch))
        accuracy_per_class_in_order = compute_last_task_accuracy_per_class_in_order(net, ordered_classes,
                                                                                    test_data_loader, exp_index)

        store_analysis_results(dormant_units_results=(dormant_units_prop_before, dormant_units_prop_after),
                               stable_rank_results=(stable_rank_before, stable_rank_after),
                               accuracy_per_class_in_order=accuracy_per_class_in_order,
                               results_dir=results_dir,
                               experiment_index=exp_index)



def main():
    global DEBUG
    terminal_arguments = parse_plots_and_analysis_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)
    DEBUG = terminal_arguments.debug

    initial_time = time.perf_counter()
    analyze_results(analysis_parameters)
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()