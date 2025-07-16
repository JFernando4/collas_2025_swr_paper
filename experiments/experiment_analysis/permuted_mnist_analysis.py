import os

import numpy as np
from prettytable import PrettyTable

from mlproj_manager.file_management import read_json_file
from mlproj_manager.util import access_dict

from src.utils import parse_plots_and_analysis_terminal_arguments, plot_avg_with_shaded_region, aggregate_over_bins

DEBUG = False


def get_average_train_accuracy(results_dir: str, parameter_combinations: list[str]):
    """ Computes the average train accuracy for each parameter combination """

    average_train_accuracy = {}
    for pc in parameter_combinations:
        temp_results_dir = os.path.join(results_dir, pc)
        indices = np.atleast_1d(np.load(os.path.join(temp_results_dir, "experiment_indices.npy")))
        train_accuracy = []
        for idx in indices:
            filename = f"index-{idx}.npy"
            temp_measurement_array = np.load(os.path.join(temp_results_dir, "train_accuracy_per_checkpoint", filename))
            if DEBUG: print(f"\tParameter combination = {pc}\t{filename = }, Mean: {np.mean(temp_measurement_array):.5f}")
            train_accuracy.append(np.mean(temp_measurement_array))
        sample_size = len(train_accuracy)
        mean_train_accuracy = np.mean(train_accuracy)
        ste_train_accuracy = np.std(train_accuracy, ddof=1) / np.sqrt(sample_size)
        average_train_accuracy[pc] = (mean_train_accuracy, ste_train_accuracy, sample_size)

    return average_train_accuracy


def create_parameter_combinations_and_table(name_prototype: str, table_parameters: dict, results_dir: str):
    """
    Creates a table with the results of the parameter sweep
    Arguments:
        name_prototype: the prototype for the parameter combination, for example "stepsize-placeholder1_momentum-placeholder2"
        table_parameters: dictionary with name of placeholders and values for the placeholders, for example
                          {"placeholders": ["placeholder1", "placeholder2"], "placeholders_values": [[0.1, 0.2], [0.9, 0.99]]}
        results_dir: the directory where the results are stored
    Returns:
         table to be printed
    """

    placeholders = table_parameters["placeholders"]
    placeholders_values = table_parameters["placeholders_values"]
    precision = 2 if "precision" not in table_parameters.keys() else table_parameters["precision"]
    assert len(placeholders) == len(placeholders_values)
    if len(placeholders) > 2: raise ValueError("Can only handle one- or two-dimensional tables.")

    if len(placeholders) == 1:
        single_parameter_table(placeholders[0], placeholders_values[0], name_prototype, results_dir, precision=precision)

    if len(placeholders) == 2:
        for ph1 in placeholders_values[0]:
            temp_name_prototype = name_prototype.replace(placeholders[0], ph1)
            temp_title = placeholders[0] + f": {ph1}"
            single_parameter_table(placeholders[1], placeholders_values[1], temp_name_prototype, results_dir, temp_title, precision=precision)


def single_parameter_table(placeholder: str, placeholder_values: list[str], name_prototype: str, results_dir: str,
                           title: str = None, precision: int = 2):
    table = PrettyTable([placeholder] + [str(val) for val in placeholder_values],
                        title=None if title is None else title, header_style=None if title is None else "title")
    sample_average_row, sample_ste_row, sample_size_row = ["Sample Avg"], ["Sample STE"], ["Sample Size"]
    for ph1 in placeholder_values:
        temp_parameter_combination = name_prototype.replace(placeholder, ph1)
        temp_acc = get_average_train_accuracy(results_dir, [temp_parameter_combination])
        temp_avg, temp_ste, temp_sample_size = temp_acc[temp_parameter_combination]
        sample_average_row.append(f"{round(temp_avg * 100, precision)}")
        sample_ste_row.append(f"{round(temp_ste * 100, precision)}")
        sample_size_row.append(f"{temp_sample_size}")
    table.add_row(sample_average_row)
    table.add_row(sample_ste_row)
    table.add_row(sample_size_row)
    print(table)


def plot_sensitivity_curve(name_prototype: str, table_parameters: dict, results_dir: str, plot_parameters: dict,
                           plot_dir: str, save_plots: bool, plot_name_prefix: str, num_samples: int):
    """
        Creates a sensitivity plot with the results of the parameter sweep
        Arguments:
            name_prototype: the prototype for the parameter combination, for example "stepsize-placeholder1_momentum-placeholder2"
            table_parameters: dictionary with name of placeholders and values for the placeholders, for example
                              {"placeholders": ["placeholder1", "placeholder2"], "placeholders_values": [[0.1, 0.2], [0.9, 0.99]]}
            results_dir: the directory where the results are stored
            other arguments: see plot_avg_with_shaded_region in src.utils.analysis_and_plotting
        Returns:
             None but displays or saves a plot
        """

    placeholders = table_parameters["placeholders"]
    placeholders_values = table_parameters["placeholders_values"]
    if len(placeholders) == 1:
        placeholders = ["", placeholders[0]]
        placeholders_values = [[""], placeholders_values[0]]
    assert len(placeholders) == len(placeholders_values)
    if len(placeholders)  != 2: raise ValueError("Can only handle two-dimensional tables.")

    results_avg, results_lows, results_highs = {}, {}, {}
    for ph1 in placeholders_values[0]:
        temp_label = placeholders[0] + f": {ph1}"
        averages, lows, highs = [], [], []
        for ph2 in placeholders_values[1]:
            temp_parameter_combination = name_prototype.replace(placeholders[0], ph1).replace(placeholders[1], ph2)
            temp_acc = get_average_train_accuracy(results_dir, [temp_parameter_combination])

            temp_avg, temp_ste, temp_ns = temp_acc[temp_parameter_combination]
            if temp_ns != num_samples:
                raise ValueError(f"Samplesize mismatch for parameter combination: {temp_parameter_combination}."
                                 f" Expected {num_samples} but got {temp_ns}.")
            averages.append(temp_avg); lows.append(temp_avg - temp_ste); highs.append(temp_avg + temp_ste)
        results_avg[temp_label] = averages
        results_lows[temp_label] = lows
        results_highs[temp_label] = highs

    plot_avg_with_shaded_region(results_avg=results_avg, results_low=results_lows, results_high=results_highs,
                                x_axis=placeholders_values[1], plot_parameters=plot_parameters, plot_dir=plot_dir,
                                measurement_name="sensitivity_curve", save_plots=save_plots,
                                plot_name_prefix=plot_name_prefix, num_samples=num_samples)



def get_results_data(results_dir: str, measurement_name: str, parameter_combination: list[str], bin_size=1,
                     convert_to_np_array=True):

    results = {}
    for pc in parameter_combination:
        if DEBUG:
            print(f"Parameter combination {pc}")
        temp_results_dir = os.path.join(results_dir, pc)
        indices = np.load(os.path.join(temp_results_dir, "experiment_indices.npy"))
        measurement_dir = os.path.join(temp_results_dir, measurement_name)

        results[pc] = []
        for idx in indices:
            filename = f"index-{idx}.npy"
            try:
                temp_measurement_array = np.load(os.path.join(measurement_dir, filename))
            except EOFError:
                if DEBUG:
                    print(f"\n{filename = }\nParameter combination = {pc}\nMeasurement = {measurement_name}")
                    print(f"\n{results_dir = }\n")
                raise EOFError
            results[pc].append(aggregate_over_bins(temp_measurement_array, bin_size, "mean"))
            if DEBUG:
                print(f"\tIndex: {idx}\tAverage Measurement: {np.mean(results[pc][-1]):.5f}")
        if convert_to_np_array:
            results[pc] = np.array(results[pc])

    return results


def compute_difference_in_loss_after_reinitialization(results_dir: str, parameter_combinations: list[str]):
    """ Computes the difference in loss before and after reinitialization """

    loss_before = get_results_data(results_dir, "loss_before_topology_update", parameter_combinations, convert_to_np_array=False)
    loss_after = get_results_data(results_dir, "loss_after_topology_update", parameter_combinations, convert_to_np_array=False)

    for pc in parameter_combinations:
        print(f"\t{pc}")
        average_difference = []
        for i in range(len(loss_before[pc])):
            min_length = min(len(loss_before[pc][i]), len(loss_after[pc][i]))
            average_difference.append(np.average(np.abs(loss_after[pc][i][:min_length] - loss_before[pc][i][:min_length])))
        total_average = np.average(average_difference)
        ste_average_difference = np.std(average_difference, ddof=1) / np.sqrt(len(average_difference))
        print(f"\t\tAverage Difference: {total_average:.6f}")
        print(f"\t\tStandard Error of Difference: {ste_average_difference:.6f}")
        print(f"\t\tNumber of Samples: {len(average_difference)}")


def compute_difference_statistics_after_reinitialization(results_dir: str, parameter_combinations: list[str]):
    """ Computes the difference in loss before and after reinitialization """

    loaded_results = {
        "layer_1": {
            "average": get_results_data(results_dir, "change_in_average_activation_layer_1", parameter_combinations, convert_to_np_array=False),
            "std": get_results_data(results_dir, "change_in_std_activation_layer_1", parameter_combinations, convert_to_np_array=False)
        },
        "layer_2": {
            "average": get_results_data(results_dir, "change_in_average_activation_layer_2", parameter_combinations, convert_to_np_array=False),
            "std": get_results_data(results_dir, "change_in_std_activation_layer_2", parameter_combinations, convert_to_np_array=False)
        },
        "layer_3": {
            "average": get_results_data(results_dir, "change_in_average_activation_layer_3", parameter_combinations, convert_to_np_array=False),
            "std": get_results_data(results_dir, "change_in_std_activation_layer_3", parameter_combinations, convert_to_np_array=False)
        }
    }

    for pc in parameter_combinations:
        print(f"\t{pc}")
        for l in range(3):
            print(f"\t\tLayer {l + 1}")
            for stat in ["average", "std"]:
                print(f"\t\t\t{stat}")
                list_of_averages = [np.average(loaded_results[f"layer_{l + 1}"][stat][pc][i]) for i in range(len(loaded_results[f"layer_{l + 1}"][stat][pc]))]
                total_average = np.average(list_of_averages)
                ste_average_difference = np.std(list_of_averages, ddof=1) / np.sqrt(len(list_of_averages))
                print(f"\t\t\t\tAverage Difference: {total_average:.6f}")
                print(f"\t\t\t\tStandard Error of Difference: {ste_average_difference:.6f}")
                print(f"\t\t\t\tNumber of Samples: {len(list_of_averages)}")


def analyse_results(analysis_parameters: dict, save_plots: bool = True):

    results_dir = analysis_parameters["results_dir"]
    parameter_combinations = analysis_parameters["parameter_combinations"]
    summary_names = analysis_parameters["summary_names"]
    plot_dir = access_dict(analysis_parameters, "plot_dir", default="")
    bin_sizes = access_dict(analysis_parameters, "bin_sizes", default=[1] * len(summary_names), val_type=list)
    table_parameters = access_dict(analysis_parameters, "table_parameters", default={}, val_type=dict)
    plot_parameters = access_dict(analysis_parameters, "plot_parameters", default={}, val_type=dict)
    plot_name_prefix = access_dict(analysis_parameters, "plot_name_prefix", default="", val_type=str)

    for sn, bs in zip(summary_names, bin_sizes):

        if sn == "difference_in_loss_after_reinitialization":
            print(f"Summary name: {sn}")
            compute_difference_in_loss_after_reinitialization(results_dir, parameter_combinations)
        elif sn == "difference_statistics_after_reinitialization":
            compute_difference_statistics_after_reinitialization(results_dir, parameter_combinations)
        elif sn == "parameter_sweep":
            assert len(parameter_combinations) == 1
            create_parameter_combinations_and_table(parameter_combinations[0], table_parameters, results_dir)
        elif sn == "sensitivity_curve":
            assert len(parameter_combinations) == 1
            plot_sensitivity_curve(parameter_combinations[0], table_parameters, results_dir, plot_parameters,
                                   plot_dir, save_plots, plot_name_prefix, analysis_parameters["sample_size"])
        elif sn == "average_accuracy":
            average_train_accuracy = get_average_train_accuracy(results_dir, parameter_combinations)
            print("\nAverage Train Accuracy")
            for pc, (average, ste, sample_size) in average_train_accuracy.items():
                print(f"\t{pc}\n\tAverage: {average:.4f}\n\tStandard Error: {ste:.4f}\tSample Size: {sample_size}")
        else:
            results_data = get_results_data(results_dir, sn, parameter_combinations, bin_size=bs)
            plot_avg_with_shaded_region(results_data=results_data, plot_parameters=plot_parameters, plot_dir=plot_dir,
                                        measurement_name=sn, save_plots=save_plots, plot_name_prefix=plot_name_prefix)


if __name__ == "__main__":

    terminal_arguments = parse_plots_and_analysis_terminal_arguments()
    analysis_parameters = read_json_file(terminal_arguments.config_file)

    DEBUG = terminal_arguments.debug

    analyse_results(analysis_parameters, save_plots=terminal_arguments.save_plot)
