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
function Rot-12k_LB_ORN_8_Align {
    define dataset "MNIST-rot-12k"
    define model "lb-orn-12k"
    define savePath "logs/Rot-12k_LB_ORN_8_Align"
    define note "Rot-12k_LB_ORN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

function Rot-12k_LB_ORN_8_Pool {
    define dataset "MNIST-rot-12k"
    define model "lb-orn-12k"
    define savePath "logs/Rot-12k_LB_ORN_8_Pool"
    define note "Rot-12k_LB_ORN_8_Pool"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=false,useORPooling=true}"
    th train.lua
}

# run tasks
PID=""
(Rot-12k_LB_ORN_8_Align) &
PID="$PID $!"
waiting $PID
