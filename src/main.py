import argparse
from tools import load_tasks, perform_test, save_output

def arg_inputs():
    # initialize parser
    my_parser = argparse.ArgumentParser(description="A script that performs color knowledge tests on different LM's")
    # add arguments
    my_parser.add_argument("-lm",
                        "--language_model",
                        type=str,
                        required=True,
                        help="the name of an LM to test")
    my_parser.add_argument("-t",
                        "--task_file",
                        type=str,
                        required=True,
                        help="The name of the file containing the tasks, should end with .txt with each line having one task")
    my_parser.add_argument("-o",
                        "--output_filename",
                        type=str,
                        required=True,
                        help="the filename of the outputs, has to end with.csv")
    args = my_parser.parse_args()
    # return list of arguments
    return args

def main():
    arguments = arg_inputs()
    output = perform_test(model = arguments.language_model, 
                          file_name=arguments.task_file)

    if type(output) == str:
        print(output)

    else:
        save_output(output = output, output_filename=arguments.output_filename)

if __name__ == "__main__":
    main()