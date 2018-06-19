--  Oriented Response Network
--  This is an implementation of the oriented response networks described in:
--  "Oriented Response Networks", https://arxiv.org/pdf/1701.01833
--  authored by Yanzhao Zhou, Qixiang Ye, Qiang Qiu and Jianbin Jiao 

--  Acknowledgement: WRN(github.com/szagoruyko/wide-residual-networks)
--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************
local nn = require 'nn'
local utils = paths.dofile'utils.lua'
require 'cudnnorn'

local scale_factor = 4
local nOrientation = 8
local ConvCounter = 0

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
end
local SBatchNorm = function (nFeature)
   local ORNFeature = nFeature / scale_factor * nOrientation
   return nn.SpatialBatchNormalization(ORNFeature)
end
local View = function (nFeature)
   local ORNFeature = nFeature / scale_factor * nOrientation
   return nn.View(ORNFeature)
end
local Linear = function (nFeature, nOutput)
   local ORNFeature = nFeature / scale_factor * nOrientation
   return nn.Linear(ORNFeature, nOutput)
end
local Align = nn.ORAlign
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.num_classes)
   assert(opt and opt.widen_factor)

   local function Dropout()
      return nn.Dropout(opt and opt.dropout or 0,nil,true)
   end

   local depth = opt.depth

   local blocks = {}
   
   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()     

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end
     
      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
     
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(Align(nOrientation))
      model:add(View(nStages[4]):setNumInputDims(3))
      model:add(Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)
   
   return model
end

return createModel