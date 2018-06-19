-- This version  LBORN with no rotation in the 1*1 conv layer,
-- learn the orientation and filter weight simutanously

require 'nn'
local utils = paths.dofile '../utils.lua'

Conv = nn.SpatialConvolutionMM

-- The base LBoM donot deploy ARF to the second conv layer
return function (opt)
    assert(opt.orientation, 'missing orientation')
    -- Local Binary ORN
    kSparsity = 0.5
    nChannels = 128 * opt.orientation
    local model = nn.Sequential()

    -- extend input to an omnidirectional tensor map
    if opt.dataset == "MNIST-rot-12k" then
        model:add(nn.View(-1, 28, 28))
    else
        model:add(nn.View(-1, 32, 32))
    end
    -- creates an output where the input is replicated nFeature times 
    -- along dimension dim (default 1)
    model:add(nn.Replicate(opt.orientation, 2))

    -- feature learning
    -- base LBConv-1 module
    model:add(nn.LBORConv(1, 128, opt.orientation, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.ReLU())
    model:add(Conv(nChannels, 10 * opt.orientation, 1, 1))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 

    -- base LBConv-2 module
    if opt.dataset == "MNIST-rot-12k" then
        model:add(nn.LBORConv(10, 128, opt.orientation, 3, 3, 1, 1, 1, 1, kSparsity))
    else
        model:add(nn.LBORConv(10, 128, opt.orientation, 3, 3, 1, 1, 0, 0, kSparsity))
    end
    
    model:add(nn.ReLU())
    model:add(Conv(nChannels, 20 * opt.orientation, 1, 1))           
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-3 module
    model:add(nn.LBORConv(20, 128, opt.orientation, 3, 3, 1, 1, 1, 1, kSparsity))
    model:add(nn.ReLU())
    model:add(Conv(nChannels, 40 * opt.orientation, 1, 1))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- base LBConv-4 module
    model:add(nn.LBORConv(40, 128, opt.orientation, 3, 3, 1, 1, 0, 0, kSparsity))
    model:add(nn.ReLU())
    model:add(Conv(nChannels, 80 * opt.orientation, 1, 1))

    -- rotation invariant encoding
    local nFeatureDim = nil
    if opt.useORAlign then
        -- ORAlign
        model:add(nn.ORAlign(opt.orientation))
        nFeatureDim = 80 * opt.orientation
    elseif opt.useORPooling then
        -- ORPooling
        model:add(nn.View(80, opt.orientation))
        model:add(nn.Max(2, 2))
        nFeatureDim = 80
    else
        -- None
        nFeatureDim = 80 * opt.orientation
    end

    -- classifier
    model:add(nn.View(nFeatureDim))              
    model:add(nn.Linear(nFeatureDim, 1024)) 
    model:add(nn.ReLU())    
    model:add(nn.Dropout(0.5))
    -- change for Outex
    model:add(nn.Linear(1024, 24))     

    return model
end
