#!/bin/bash

#
# Copyright (c) 2009-2018 Brandon Haworth, Glen Berseth, Muhammad Usman, Mubbasir Kapadia, Shawn Singh, Petros Faloutsos, Glenn Reinman
# See license.txt for complete license.
#

# this sciprt is designed to package a steersuite for release.


# example usage
# ./bundleCode.sh 
os="linux"
tarOptions="--exclude='.svn/' --exclude='.git/' --exclude-vcs"

tarName="smirl.tar"
rm $tarName # if old files
#	tar $tarOptions -cvf $tarName ./build/win32/
tar $tarOptions -cvf $tarName ./configs
tar $tarOptions -cvf $tarName ./util
tar $tarOptions -rvf $tarName ./launchers/config.py
tar $tarOptions -rvf $tarName ./launchers/config.py
tar $tarOptions -rvf $tarName ./scripts/
tar $tarOptions -rvf $tarName ./surprise/
tar $tarOptions -rvf $tarName ./Dockerfile
tar $tarOptions -rvf $tarName ./README.md/
tar $tarOptions -rvf $tarName ./requirements.txt
tar $tarOptions -rvf $tarName ./setup.py
tar $tarOptions -rvf $tarName ./Singularity
tar $tarOptions -rvf $tarName ./build_push_docker.sh
tar $tarOptions -rvf $tarName ../doodad

# tar $tarOptions -rvf $tarName paramBlendingDemo.sh
gzip -f --best $tarName
