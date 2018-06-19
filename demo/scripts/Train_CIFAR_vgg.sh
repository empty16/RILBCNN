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

function CIFAR10_OR_VGG {
    define dataset "CIFAR-10"
    define model "or-vgg"
    define savePath "logs/CIFAR10_OR_VGG"
    define note "CIFAR10_OR_VGG"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function CIFAR10_ORN {
    define dataset "CIFAR-10"
    define model "ORN"
    define savePath "logs/CIFAR10_ORN"
    define note "CIFAR10_ORN"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

# tasks
function CIFAR10_LB_OR_VGG {
    define dataset "CIFAR-10"
    define model "lb-or-vgg"
    define savePath "logs/CIFAR10_LB_OR_VGG"
    define note "CIFAR10_LB_OR_VGG"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function CIFAR100_LB_OR_VGG {
    define dataset "CIFAR-100"
    define model "lb-or-vgg"
    define savePath "logs/CIFAR100_LB_OR_VGG"
    define note "CIFAR100_LB_OR_VGG"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

function CIFAR10_VGG {
    define dataset "CIFAR-10"
    define model "vgg"
    define savePath "logs/CIFAR10_VGG"
    define note "CIFAR10_VGG"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

# run tasks
PID=""
(CIFAR10_LB_OR_VGG) &
#(CIFAR10_LB_OR_VGG; CIFAR100_LB_OR_VGG) &
PID="$PID $!"
waiting $PID
