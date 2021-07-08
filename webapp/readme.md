### This repository is aimed at deploying a Flask webapp able to predict maximum emission wavelengths based on SMILES of organic molecule. The target platform is RaspberryPi

## Setting up
Since RaspiOS has outdated RDKit module, then it is necessary to install some other OS. Ubuntu Server is a good choice. To set it up:

1. Install [RaspberryPi Imager](https://www.raspberrypi.org/blog/raspberry-pi-imager-imaging-utility/) and use it to install Ubuntu Server on SD card.
2. Plug SD card to your RPi and follow instructions. Default user and password are both "ubuntu". In case of problems with booting it may be necessary to plug off all USB devices (I encountered such issue). When the boot starts plug in your keyboard.
3. Update system with following commands:

```
sudo apt update
sudo apt upgrade
```

4. 
