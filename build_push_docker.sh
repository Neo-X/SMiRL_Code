#!/bin/bash

docker build -f Dockerfile -t smirl:latest .
docker tag smirl:latest $USER/smirl:latest
docker push $USER/smirl:latest

