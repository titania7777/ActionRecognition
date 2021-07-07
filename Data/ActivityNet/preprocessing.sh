#!/bin/bash

# Version 1.3
tar -xzvf v1-3_train_val.tar.gz
mv ./v1-3/train_val/* ./v1-3/
rm -rf ./v1-3/train_val/
mv ./v1-3/ ./videos/