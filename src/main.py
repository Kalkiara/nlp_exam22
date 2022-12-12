import argparse
from tools import load_tasks, perform_test, save_output

def main(model, task_file, output_filename):
    tasks = load_tasks(task_file)
    output = perform_test(tasks, model)
    if type(output) == str:
        print(output)
    else:
        save_output(output, output_filename)

if __name__ == "__main__":
    main(model = 'bert_large', 
        task_file='tasks_bert.txt', 
        output_filename='output.csv')