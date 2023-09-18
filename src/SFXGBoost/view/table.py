# creates tables for latex / markdown (?)


def create_latex_table_1(h_axis, data, name):
    latex_table = "\\begin{tabular}{| "+ "|".join(["c" for _ in h_axis]) + "|}\n"
    latex_table = "\\hline\n" + " & ".join(h_axis) + "\\\\\n\\hline\n"

    for row in data:
        latex_table += " & ".join(row) + "\\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}"
    output_filename = name + ".txt"
    
    with open(output_filename, 'w') as f:
        f.write(latex_table)
        
def table_experiment1(all_data):

    # all_data[targetArchitecture][dataset][parameter_name][val] = data
    # create_latex_table_1(h_axis=)
    # creates a table for every possible parameter_name
    pass