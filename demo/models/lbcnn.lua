require 'nn'
local utils = paths.dofile '../utils.lua'

return function (opt)
    -- Local Binary CNN
    kSparsity = 0.5
    local model = nn.Sequential()
    -- feature learning
    -- base LBConv-1 module
    model:add(nn.LBConv(1, 512, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(512, 80, 1, 1))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 

    -- base LBConv-2 module
    model:add(nn.LBConv(80, 512, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(512, 160, 1, 1))           
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-3 module
    model:add(nn.LBConv(160, 512, 3, 3, 1, 1, 1, 1, kSparsity))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(512, 320, 1, 1))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-4 module
    model:add(nn.LBConv(320, 512, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(512, 640, 1, 1))

    -- classifier
    model:add(nn.View(640))              
    model:add(nn.Linear(640, 1024)) 
    model:add(nn.ReLU())    
    model:add(nn.Dropout(0.5))
    -- change for Outex 24
    model:add(nn.Linear(1024, 10))     

    return model
end