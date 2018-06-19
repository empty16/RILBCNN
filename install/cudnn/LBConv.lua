-- needs global variable kSparsity to be defined

cudnn = require 'cudnn'
local LBConv, parent = torch.class('cudnn.LBConv', 'nn.LBConv')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

function LBConv:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    -- self.padW = padW or 0
    -- self.padH = padH or 0
    self:reset()
    -- should nil for serialization, the reset will still work
    -- self.reset = nil

end

function LBConv:reset()
  local numElements = self.nInputPlane * self.nOutputPlane * self.kW * self.kH
  self.weight = torch.CudaCharTensor(self.nOutputPlane, self.nInputPlane, self.kW, self.kH):fill(0)
  self.weight = torch.reshape(self.weight, numElements)
  local index = torch.Tensor(torch.floor(kSparsity * numElements)):random(numElements)
  for i = 1,index:numel() do
    self.weight[index[i]] = torch.bernoulli(0.5) * 2 - 1
  end
  self.weight = torch.reshape(self.weight, self.nOutputPlane, self.nInputPlane, self.kW,self.kH)

  self.bias = nil
  self.gradBias = nil 
  self.gradWeight = torch.CudaCharTensor(self.nOutputPlane, self.nInputPlane, self.kW, self.kH):fill(0)   
end

--backward function
function LBConv:accGradParameters(input, gradOutput, scale)
end

--backward function frezze layer parameters
function LBConv:updateParameters(learningRate)
end