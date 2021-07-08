### This repository is aimed at deploying a Flask webapp able to predict maximum emission wavelengths based on SMILES of organic molecule. The target platform is RaspberryPi

## Setting up
Since RaspiOS has outdated RDKit module, then it is necessary to install some other OS. Ubuntu Server is a good choice. To set it up:

1. Install [RaspberryPi Imager](https://www.raspberrypi.org/blog/raspberry-pi-imager-imaging-utility/) and use it to install Ubuntu Server on SD card.
2. Plug SD card to your RPi and follow instructions. Default user and password are both "ubuntu". In case of problems with booting it may be necessary to plug off all USB devices (I encountered such issue). When the boot starts plug in your keyboard.
3. Update system with following commands:

``` bash
sudo apt update
sudo apt upgrade
```

4. Install necessary packages:

```
sudo apt install python3-pip python3-rdkit
```

5. Install necessary Python packages:

```
python3 -m pip install flask numpy pandas Flask-WTF scikit-learn 
```
