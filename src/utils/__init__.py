from .evaluation_functions import compute_accuracy_from_batch, compute_average_gradient_magnitude, set_random_seed, \
    compute_matrix_rank_summaries, compute_average_weight_magnitude, bootstrapped_return, compute_prop_dead_units
from .data_management import (get_cifar_data, subsample_cifar_data_set, get_tiny_imagenet_data, get_data_loader,
                              set_tiny_imagenet_shape_and_normalization_transforms, set_tiny_imagenet_data_augmentation_transforms)
from .experiment_utils import parse_terminal_arguments, parse_plots_and_analysis_terminal_arguments
from .cifar100_experiment_utils import *
from .incremental_cifar_experiment_base import IncrementalCIFARExperimentBase
from .permuted_mnist_experiment_utils import compute_dead_units_proportion
from .analysis_and_plotting import COLOR_DICT, aggregate_over_bins, plot_results, plot_avg_with_shaded_region
