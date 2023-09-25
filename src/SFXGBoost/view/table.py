# creates tables for latex / markdown (?)


def create_latex_table_tmp(h_axis, data, name):
    latex_table = "\\begin{tabular}{| "+ "|".join(["c" for _ in h_axis]) + "|}\n"
    latex_table = "\\hline\n" + " & ".join(h_axis) + "\\\\\n\\hline\n"

    for row in data:
        latex_table += " & ".join(row) + "\\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}"
    output_filename = name + ".txt"
    
    with open(output_filename, 'w') as f:
        f.write(latex_table)
        
def create_latex_table_1(all_data, to_be_tested, metrics, name_model, datasets, destination="Table/experiment_1.txt"):
    
    # name_model = "FederBoost-central"
    # to_be_tested = {"gamma": [0, 0.1, 0.25, 0.5, 0.75, 1, 5, 10], # [0,inf] minimum loss for split to happen default = 0
    #             "max_depth": [5, 8, 12],
    #             "max_trees": [5, 10, 20, 30, 50, 100, 150],
    #             "training_size": [1000, 2000, 5000, 10_000, 30_000],
    #             "alpha": [0, 0.1, 0.25, 0.5, 0.75, 1, 10],  # [0, inf] L1 regularisation default = 0
    #             "lam":   [0, 0.1, 0.25, 0.5, 0.75, 1, 10],  # L2 regularisation [0, inf] default = 1
    #             "eta":   [0, 0.1, 0.25, 0,5 ,0.75, 1]  # learning rate [0,1] default = 0.3
    #             }
    # metrics = "overfitting, acc"
    num_columns = (len(metrics) * len(datasets)) + 1
    for param_name in to_be_tested.keys():
        latex_table = "\\begin{*table}{|*{" + str(num_columns) + "}{c|}\n"
        latex_table += "\\centering\n"
        latex_table += "\\hline\\rowcolor{gray!50}"
        latex_table += "\\cellcolor{gray!80} "+ param_name + " & ".join(["\\multicolumn{" + str(len(metrics)) + "}{c|}{"+dataset+"}" for dataset in datasets]) +"\\\\\\hline \n"
        repeated = [metrics for _ in range(len(datasets))]
        latex_table += "&" + " & ".join( [ metric for metric in repeated])
        for val in to_be_tested[param_name]:
            latex_table += val + " & " + " & ".join(
                [" & ".join([result for result in [all_data[name_model][param_name][dataset][val][metric] for metric in metrics]
                            ])for dataset in datasets]) + "\\\\hline\n"
        latex_table += "\\end{tabular} \n"
        latex_table += "\\caption{" + name_model + "'s attack metrics on "+ param_name + ".}\n"
        latex_table += "\\label{tab:experiment1_"+ param_name+ "}\n"
        latex_table += "\\end{table*}\n"

        tmp_destination = destination.replace(".txt", f"{param_name}.txt")
        with open(tmp_destination, 'w') as file:
            file.write(latex_table)

    

def table_experiment1(all_data):

    # all_data[targetArchitecture][parameter_name][dataset][val] = data
    # create_latex_table_1(h_axis=)
    # creates a table for every possible parameter_name
    pass

