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
init maxEpoch 200
init learningRateDecayRatio 0.5
init removeOldCheckpoints false
init optimMethod "adadelta"
init batchSize 128
init gpuDevice "{1}"

# Ori tasks
function Ori_ORN_8_None {
    define dataset "MNIST-ori"
    define model "ORN"
    define savePath "logs/Ori_ORN_8_None"
    define note "Ori_ORN_8_None"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=false}"
    th train.lua
}

function Ori_ORN_8_Pooling {
    define dataset "MNIST-ori"
    define model "ORN"
    define savePath "logs/Ori_ORN_8_Pooling"
    define note "Ori_ORN_8_Pooling"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=false,useORPooling=true}"
    th train.lua
}

function Ori_ORN_8_Align {
    define dataset "MNIST-ori"
    define model "ORN"
    define savePath "logs/Ori_ORN_8_Align"
    define note "Ori_ORN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=4,useORAlign=true,useORPooling=false}"
    th train.lua
}

# Rot tasks
function Rot_ORN_8_None {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_8_None"
    define note "Rot_ORN_8_None"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=false,useORPooling=false}"
    th train.lua
}

function Rot_ORN_8_Pooling {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_8_Pooling"
    define note "Rot_ORN_8_Pooling"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=false,useORPooling=true}"
    th train.lua
}

function Rot_ORN_8_Align {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_8_Align"
    define note "Rot_ORN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

# Ori2Rot tasks
function Ori2Rot_ORN_8_None {
    define dataset "MNIST-ori2rot"
    define model "ORN"
    define savePath "logs/Ori2Rot_ORN_8_None"
    define note "Ori2Rot_ORN_8_None"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=false,useORPooling=false}"
    th train.lua
}

function Ori2Rot_ORN_8_Pooling {
    define dataset "MNIST-ori2rot"
    define model "ORN"
    define savePath "logs/Ori2Rot_ORN_8_Pooling"
    define note "Ori2Rot_ORN_8_Pooling"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=false,useORPooling=true}"
    th train.lua
}

function Ori2Rot_ORN_8_Align {
    define dataset "MNIST-ori2rot"
    define model "ORN"
    define savePath "logs/Ori2Rot_ORN_8_Align"
    define note "Ori2Rot_ORN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

# run tasks
PID=""
(Ori_ORN_8_None; Ori_ORN_8_Pooling; Ori_ORN_8_Align;
 Rot_ORN_8_None; Rot_ORN_8_Pooling; Rot_ORN_8_Align;
 Ori2Rot_ORN_8_None; Ori2Rot_ORN_8_Pooling; Ori2Rot_ORN_8_Align) &
PID="$PID $!"
waiting $PID
