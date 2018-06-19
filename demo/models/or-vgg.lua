-- This is a modified version of VGG network in
-- https://github.com/szagoruyko/cifar.torch
-- Modifications:
--  * removed dropout
--  * last nn.Linear layers substituted with convolutional layers
--    and avg-pooling
require 'nn'
local utils = paths.dofile'utils.lua'
require 'cudnnorn'
local nOrientation = 8
local ConvCounter = 0
local scale_factor = 4

-- function Convolution 
local Convolution = function (...)
   local arg = {...}
   local ARF
   if ConvCounter > 0 then
      arg[1] = arg[1] / scale_factor
      arg[2] = arg[2] / scale_factor
      ARF = nOrientation
   else
      arg[2] = arg[2] / scale_factor
      ARF = {1, nOrientation}
   end
   ConvCounter = ConvCounter + 1
   table.insert(arg, 3, ARF)
   return nn.ORConv(unpack(arg))
   -- unpack(arg): return all the parameters
end

-- function View
local View = function (nFeature)
   local ORNFeature = nFeature / scale_factor * nOrientation
   return nn.View(ORNFeature)
end

-- function Linear
local Linear = function (nFeature, nOutput)
   local ORNFeature = nFeature / scale_factor * nOrientation
   return nn.Linear(ORNFeature, nOutput)
end

-- function BN
local SBatchNorm = function (nFeature)
   local ORNFeature = nFeature / scale_factor * nOrientation
   return nn.SpatialBatchNormalization(ORNFeature)
end

return function (opt)
   local model = nn.Sequential()

   -- building block
   local function Block(nInputPlane, nOutputPlane)
      model:add(Convolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
      model:add(SBatchNorm(nOutputPlane, 1e-3))
      model:add(nn.ReLU())
      return model
   end

   local function MP()
      model:add(nn.SpatialMaxPooling(2,2, 2,2):ceil())
      return model
   end

   local function Group(ni, no, N, f)
      -- the priority of the logical operator (and > or)
      for i=1, N do
         Block(i == 1 and ni or no, no)
      end
      -- proper tail call
      if f then f() end
   end

   -- feature learning
   -- model:add(Convolution(3,8, 3,3, 1,1, 1,1)) -- one conv at the beginning (spatial size: 32x32)
   -- model:add(nn.ReLU()) -- first conv layer
   -- model:add(nn.ORConv(8, 8, nOrientation, 3,3, 1,1, 1,1))
   -- model:add(nn.ReLU())
   -- model:add(nn.SpatialMaxPooling(2,2, 2,2):ceil())

   Group(3, 64, 2, MP)
   Group(64, 128, 2, MP)
   Group(128, 256, 4, MP)
   Group(256, 512, 4, MP)
   Group(512, 512, 4)

   model:add(nn.SpatialAveragePooling(2,2, 2,2):ceil())
   model:add(nn.ORAlign(nOrientation))

   -- classifier
   model:add(View(512):setNumInputDims(3))
   model:add(Linear(512, opt and opt.numClasses or 10))

   -- initailization
   utils.FCinit(model)
   utils.testModel(model)
   utils.MSRinit(model)

   return model
end