# Experiment on Google Cloud Platform

The commands in this manual make use of the `gcloud` cli tool.
Make sure that this is installed on your machine.
The commands are tested on Ubuntu 20.04, but it should work on most Unix systems and on Windows WSL.

The console can be found at [https://console.cloud.google.com/](https://console.cloud.google.com/)

* Enable the Compute Engine in the console.

## One Client example
This example creates one federator and one client instance.

Start by creating two VM instances on the `Compute Engine`.

1. Create the Federator VM instance with the following properties:

| Property        | Value        |
| ------------- |:-------------:| 
| Name      | `federator` |
| Region     | `europe-west4`     |
| Machine Family | General Compute      |
| Machine series | `N1` |
| Machine Type | `g1-small`
| Boot disk | Ubuntu 20.04 |
| Boot disk type | Balanced Persistent Disk |
| Disk Size|  10 GB |
| Firewall | `Allow HTTP traffic` and `Allow HTTPS traffic ` |
| Networking External ip | Ephemeral |

2. Create the Client VM instance with the following properties:

| Property        | Value        |
| ------------- |:-------------:| 
| Name      | `client1` |
| Region     | `europe-west4`     |
| Machine Family | General Compute      |
| Machine series | `N1` |
| Machine Type | `g1-small`
| Boot disk | Ubuntu 20.04 |
| Boot disk type | Balanced Persistent Disk |
| Disk Size|  10 GB |
| Firewall | `Allow HTTP traffic` and `Allow HTTPS traffic ` |
| Networking External ip | Ephemeral |

3. Once the VMs have started, SSH into the VMs and follow the next steps:

   * `sudo apt update`
   * `git clone https://github.com/bacox/fltk.git fltk`
   * `cd fltk`
   * `python3 setup.py install`
   
Make sure you have done this at both the `federator` and the `client1` VM instance.

4. Get the ip address of the `federator` instance:

   * SSH into the `federator` instance
   * Execute `ip a` in the shell
   * The ip address is probally on NIC `ens4`
   
5. On the `federator` instance (via SSH) start the system by executing from the `fltk` folder:
   ```bash
   python3 -m fltk single configs/experiment.yaml --rank=0 -–nic=<NIC from step 4> -–host=<ip fromstep 4>
   ```
   
6. On the `client1` instance (via SSH) start the system by executing from the `fltk` folder:
   ```bash
   python3 -m fltk single configs/experiment.yaml --rank=1 -–nic=<NIC from step 4> -–host=<ip fromstep 4>
   ```
   
   