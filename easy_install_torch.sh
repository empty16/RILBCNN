#!/usr/bin/env bash

echo "step 1. install dependencies"
luarocks install torchnet
luarocks install optnet

echo "step 2. clone torch implementation of ORN"
git clone https://github.com/ZhouYanzhao/ORN.git -b torch --single-branch ORN.torch
cd ORN.torch
export DIR=$(pwd)

echo "step 3. install ORN"
cd $DIR/install && bash install.sh

echo "step 4. unzip dataset"
cd $DIR/demo/datasets && unzip MNIST

echo "step 5. train a conventional CNN and an ORN model on the MNIST-rot dataset"
cd $DIR/demo && bash ./scripts/Train_MNIST.sh