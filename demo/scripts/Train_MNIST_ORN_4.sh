#!/usr/bin/env bash
# CNN, ORN(4/8) @ dataset(original, rot, ori2rot)

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
init maxEpoch 200
init learningRateDecayRatio 0.5
init removeOldCheckpoints false
init optimMethod "adadelta"
init batchSize 128
init gpuDevice "{1}"

# Ori tasks
function Ori_CNN {
    define dataset "MNIST-ori"
    define model "CNN"
    define savePath "logs/Ori_CNN"
    define note "Ori_CNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function Ori_ORN_4_None {
    define dataset "MNIST-ori"
    define model "ORN"
    define savePath "logs/Ori_ORN_4_None"
    define note "Ori_ORN_4_None"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=false}"
    th train.lua
}

function Ori_ORN_4_Pooling {
    define dataset "MNIST-ori"
    define model "ORN"
    define savePath "logs/Ori_ORN_4_Pooling"
    define note "Ori_ORN_4_Pooling"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=true}"
    th train.lua
}

function Ori_ORN_4_Align {
    define dataset "MNIST-ori"
    define model "ORN"
    define savePath "logs/Ori_ORN_4_Align"
    define note "Ori_ORN_4_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=true,useORPooling=false}"
    th train.lua
}

# Rot tasks
function Ori2Rot_CNN {
    define dataset "MNIST-rot"
    define model "CNN"
    define savePath "logs/Rot_CNN"
    define note "Rot_CNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function Rot_ORN_4_None {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_4_None"
    define note "Rot_ORN_4_None"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=false}"
    th train.lua
}

function Rot_ORN_4_Pooling {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_4_Pooling"
    define note "Rot_ORN_4_Pooling"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=true}"
    th train.lua
}

function Rot_ORN_4_Align {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_4_Align"
    define note "Rot_ORN_4_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=true,useORPooling=false}"
    th train.lua
}

# Ori2Rot tasks
function Ori2Rot_CNN {
    define dataset "MNIST-ori2rot"
    define model "CNN"
    define savePath "logs/Ori2Rot_CNN"
    define note "Ori2Rot_CNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function Ori2Rot_ORN_4_None {
    define dataset "MNIST-ori2rot"
    define model "ORN"
    define savePath "logs/Ori2Rot_ORN_4_None"
    define note "Ori2Rot_ORN_4_None"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=false}"
    th train.lua
}

function Ori2Rot_ORN_4_Pooling {
    define dataset "MNIST-ori2rot"
    define model "ORN"
    define savePath "logs/Ori2Rot_ORN_4_Pooling"
    define note "Ori2Rot_ORN_4_Pooling"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=true}"
    th train.lua
}

function Ori2Rot_ORN_4_Align {
    define dataset "MNIST-ori2rot"
    define model "ORN"
    define savePath "logs/Ori2Rot_ORN_4_Align"
    define note "Ori2Rot_ORN_4_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=true,useORPooling=false}"
    th train.lua
}

# run tasks
PID=""
(Ori_CNN; Ori_ORN_4_None; Ori_ORN_4_Pooling; Ori_ORN_4_Align;
 Rot_CNN; Rot_ORN_4_None; Rot_ORN_4_Pooling; Rot_ORN_4_Align;
 Ori2Rot_CNN; Ori2Rot_ORN_4_None; Ori2Rot_ORN_4_Pooling; Ori2Rot_ORN_4_Align) &
PID="$PID $!"
waiting $PID
