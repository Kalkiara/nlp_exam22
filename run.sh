sudo apt-get update
sudo apt-get install python3-venv

python3.9 -m venv env

source ./env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python3 src/main.py -lm $1 -t $2 -o $3

deactivate