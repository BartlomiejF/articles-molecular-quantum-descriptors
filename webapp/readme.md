### This repository is aimed at deploying a Flask webapp able to predict maximum emission wavelengths based on SMILES of organic molecule. The target platform is RaspberryPi

## Setting up

1. Update
``` bash
sudo apt update
sudo apt upgrade
```

2. Install necessary packages:

```
sudo apt install python3-pip
```

3. Install necessary Python packages:

```
python3 -m pip install flask numpy pandas Flask-WTF scikit-learn 
```

4. Download this repository and QM9 database. Put the database in dbs folder.
[QM9](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv)

5. Run the training of the predictor:
```
python3
import learn
learn.train()
```

6. Run the webapp:
```
python3 main.py
```
