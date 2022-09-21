#!/bin/bash

sudo APPTAINER_NOHTTPS=1 apptainer build smirl_app.sif docker-daemon://$USER/smirl:latest
sudo chmod 777 smirl_app.sif
