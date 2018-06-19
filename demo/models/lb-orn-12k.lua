require 'nn'
local utils = paths.dofile '../utils.lua'
local nOrientation = 8
nInputPlane = 1
nChIn = 32

return function (opt)
    -- Local Binary ORN
    kSparsity = 0.5
    local model = nn.Sequential()

    -- feature learning
    -- add learnable filters at the first layer 
    model:add(nn.ORConv(nInputPlane, nChIn, {1, nOrientation}, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())

    -- base LBConv-1 module
    model:add(nn.LBORConv(nChIn, 128, nOrientation, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.SpatialBatchNormalization(128 * nOrientation, 1e-3))
    model:add(nn.ReLU())
    model:add(nn.ORConv(128, 10, nOrientation, 1, 1))
    model:add(nn.SpatialBatchNormalization(10 * nOrientation, 1e-3))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-2 module
    model:add(nn.LBORConv(10, 128, nOrientation, 3, 3, 1, 1, 1, 1, kSparsity))
    model:add(nn.SpatialBatchNormalization(128 * nOrientation, 1e-3))
    model:add(nn.ReLU())
    model:add(nn.ORConv(128, 20, nOrientation, 1, 1))
    model:add(nn.SpatialBatchNormalization(20 * nOrientation, 1e-3))           
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-3 module
    model:add(nn.LBORConv(20, 128, nOrientation, 3, 3, 1, 1, 1, 1, kSparsity))
    model:add(nn.SpatialBatchNormalization(128 * nOrientation, 1e-3))
    model:add(nn.ReLU())
    model:add(nn.ORConv(128, 40, nOrientation, 1, 1))
    model:add(nn.SpatialBatchNormalization(40 * nOrientation, 1e-3))  
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-4 module
    model:add(nn.LBORConv(40, 128, nOrientation, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.SpatialBatchNormalization(128 * nOrientation, 1e-3))
    model:add(nn.ReLU())
    model:add(nn.ORConv(128, 80, nOrientation, 1, 1))
    model:add(nn.SpatialBatchNormalization(80 * nOrientation, 1e-3))  

    -- rotation invariant encoding
    -- ORAlign
    model:add(nn.ORAlign(nOrientation))
    nFeatureDim = 80 * nOrientation

    -- classifier
    model:add(nn.View(nFeatureDim))              
    model:add(nn.Linear(nFeatureDim, 1024)) 
    model:add(nn.ReLU())    
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, 10))     

    return model
end
