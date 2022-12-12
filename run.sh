sudo apt-get update
sudo apt-get install python3-venv

python3.9 -m venv env

source ./env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python3 src/main.py -lm 'gpt2' -t 'tasks_gpt.txt' -o 'output.csv'

deactivate