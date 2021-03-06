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
init removeOldCheckpoints true
init optimMethod "adadelta"
init batchSize 128
init gpuDevice "{1}"

# tasks
function Outex_24_ori_CNN {
    define dataset "Outex-24-ori"
    define model "CNN-3"
    define savePath "logs/Outex_24_ori_CNN"
    define note "Outex_24_ori_CNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function Outex_24_ori_LBCNN {
    define dataset "Outex-24-ori"
    define model "lbcnn-3"
    define savePath "logs/Outex_24_ori_LBCNN"
    define note "Outex_24_ori_LBCNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function Outex_24_ori_ORN_8_Align {
    define dataset "Outex-24-ori"
    define model "ORN-3"
    define savePath "logs/Outex_24_ori_ORN_8_Align"
    define note "Outex_24_ori_ORN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

function Outex_24_ori_RILBCNN_8_Align {
    define dataset "Outex-24-ori"
    define model "lb-orn-3"
    define savePath "logs/Outex_24_ori_RILBCNN_8_Align"
    define note "Outex_24_ori_RILBCNN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

# run tasks
PID=""
(Outex_24_ori_CNN; Outex_24_ori_LBCNN) &
#(Outex_24_ori_RILBCNN_8_Align) &
PID="$PID $!"
waiting $PID
