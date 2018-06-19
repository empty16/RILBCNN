#!/usr/bin/env bash
# utilities
function init { 
    if [ -z "${!1}" ]; then export $1=$2; fi
}
function define { 
    export $1=$2
}
function waiting {
    for pid in "$@"
    do
        while [ -e /proc/$pid ]
        do
            sleep 1
        done
    done
}

# global settings
init enableSave true
init notify "false"
init maxEpoch 50
init learningRateDecayRatio 0.5
init removeOldCheckpoints false
init optimMethod "adadelta"
init batchSize 128
init gpuDevice "{1}"

# tasks
function CIFAR10_CNN {
    define dataset "CIFAR-10"
    define model "CNN"
    define savePath "logs/CIFAR10_CNN"
    define note "CIFAR10_CNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function CIFAR10_LB_OR_ResNet {
    define dataset "CIFAR-10"
    define model "lb-or-resnet"
    define savePath "logs/CIFAR10_LB_OR_ResNet"
    define note "CIFAR10_LB_OR_ResNet"
    define depth 20
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function CIFAR100_LB_OR_ResNet {
    define dataset "CIFAR-100"
    define model "lb-or-resnet"
    define savePath "logs/CIFAR100_LB_OR_ResNet"
    define note "CIFAR100_LB_OR_ResNet"
    define depth 20
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

# run tasks
PID=""
(CIFAR10_CNN) &
#(CIFAR10_LB_OR_VGG; CIFAR100_LB_OR_VGG) &
PID="$PID $!"
waiting $PID
