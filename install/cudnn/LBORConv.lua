cudnn = require 'cudnn'
local LBORConv, parent = torch.class('cudnn.LBORConv', 'nn.LBORConv')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

function LBORConv:__init(nInputPlane, nOutputPlane, ARF, kW, kH, dW, dH, padW, padH)
    parent.__init(self, nInputPlane, nOutputPlane, ARF, kW, kH, dW, dH, padW, padH)
    self.padW = padW or 0
    self.padH = padH or 0
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
end

-- use the bernolli distribution to initialize the weights and bias
function LBORConv:reset()
  local numElements = self.nInputPlane * self.nOutputPlane * self.nOrientation * self.kW * self.kH
  self.weight = torch.CudaCharTensor(self.nOutputPlane, self.nInputPlane, self.nOrientation, self.kW,self.kH):fill(0)
  self.weight = torch.reshape(self.weight, numElements)
  local index = torch.Tensor(torch.floor(kSparsity * numElements)):random(numElements)
  for i = 1,index:numel() do
    self.weight[index[i]] = torch.bernoulli(0.5)*2-1
  end
  self.weight = torch.reshape(self.weight, self.nOutputPlane, self.nInputPlane, self.nOrientation, self.kW, self.kH)

  self.bias = nil
  self.gradBias = nil 
  self.gradWeight = torch.CudaCharTensor(self.nOutputPlane, self.nInputPlane, self.kW, self.kH):fill(0)   
end

-- frezze the layer parmaters
function LBORConv:accGradParameters(input, gradOutput, scale)
end

-- frezze the layer parmaters
function LBORConv:updateParameters(learningRate)
end


-- if you change the configuration of the module manually, call this
function LBORConv:resetWeightDescriptors(desc)
    -- for compatibility
    assert(cudnn.typemap[torch.typename(self.weightBuffer)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, (self.nOutputPlane*self.nRotation),1,1))
    end

    self.weightBufferDesc = cudnn.setFilterDescriptor(
       { dataType = cudnn.typemap[torch.typename(self.weightBuffer)],
         filterDimA = desc or
            {(self.nOutputPlane*self.nRotation),
             (self.nInputPlane*self.nOrientation),
             self.bH, self.bW}
       }
    )

    return self
end

function LBORConv:fastest(mode)
    if mode == nil then mode = true end
    if not self.fastest_mode or self.fastest_mode ~= mode then
       self.fastest_mode = mode
       self.iDesc = nil
    end
    return self
end

function LBORConv:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iDesc = nil
    return self
end

function LBORConv:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function LBORConv:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function LBORConv:checkInputChanged(input)
    assert(input:isContiguous(),
           "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
    if not self.iSize or self.iSize:size() ~= input:dim() then
       self.iSize = torch.LongStorage(input:dim()):fill(0)
    end
    self.weightBuffer = self.weightBuffer or input.new():resize(table.unpack(self.weightBufferSize))
    self.gradWeightBuffer = self.gradWeightBuffer or input.new():resize(table.unpack(self.weightBufferSize))
    if not self.weightBufferDesc then self:resetWeightDescriptors() end
    if not self.weightBufferDesc then error "Weights not assigned!" end

    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (input:dim()==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()
       assert((self.nInputPlane*self.nOrientation) == input:size(2),
              'input has to contain: '
                 .. (self.nInputPlane*self.nOrientation)
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3)
                 .. (input:dim()>3 and ' x ' .. input:size(4) ..
                        (input:dim()==5 and ' x ' .. input:size(5) or '') or ''))
       return true
    end
    return false
end

function LBORConv:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   if LBORConv.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input:narrow(2,1,(self.nInputPlane*self.nOrientation))
        self.iDesc = cudnn.toDescriptor(input_slice)
        -- create conv descriptor
        self.padH, self.padW = self.padH or 0, self.padW or 0
        -- those needed to calculate hash
        self.pad = {self.padH, self.padW}
        self.stride = {self.dH, self.dW}

        self.convDescData = { padA = self.pad,
             filterStrideA = self.stride,
             upscaleA = {1,1},
             dataType = cudnn.configmap(torch.type(self.weightBuffer))
        }

        self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.weightBufferDesc[0], 4, oSize:data())
        oSize[2] = oSize[2]
        self.output:resize(oSize:long():storage())
        self.oSize = self.output:size()

        local output_slice = self.output:narrow(2,1,(self.nOutputPlane*self.nRotation))
        -- create descriptor for output
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        find:prepare(self, input_slice, output_slice)

        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local bH, bW = self.bH, self.bW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = (self.nInputPlane*self.nOrientation) * iH * iW
        self.output_offset = (self.nOutputPlane*self.nRotation) * oH * oW
        self.weightBuffer_offset = (self.nInputPlane*self.nOrientation) * (self.nOutputPlane*self.nRotation) * bH * bW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end

   end
   return self
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function LBORConv:updateOutput(input)
    input = makeContiguous(self, input)
    self:createIODescriptors(input)
    -- generate rotated versions of ARF on the fly
    self.ARFRotate(self, input)
    local finder = find.get()
    local fwdAlgo = finder:forwardAlgorithm(self, { self.iDesc[0], self.input_slice, self.weightBufferDesc[0],
                                                    self.weightBuffer, self.convDesc[0], self.oDesc[0], self.output_slice})
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    checkedCall(self,'cudnnConvolutionForward', cudnn.getHandle(),
                cudnn.scalar(input, 1),
                self.iDesc[0], input:data(),
                self.weightBufferDesc[0], self.weightBuffer:data(),
                self.convDesc[0], fwdAlgo,
                extraBuffer, extraBufferSize,
                cudnn.scalar(input, 0),
                self.oDesc[0], self.output:data());

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function LBORConv:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    assert(gradOutput:dim() == input:dim()-1 or gradOutput:dim() == input:dim()
              or (gradOutput:dim()==5 and input:dim()==4), 'Wrong gradOutput dimensions');
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    local bwdDataAlgo = finder:backwardDataAlgorithm(self, { self.weightBufferDesc[0], self.weightBuffer, self.oDesc[0],
                                                             self.output_slice, self.convDesc[0], self.iDesc[0], self.input_slice })
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    checkedCall(self,'cudnnConvolutionBackwardData', cudnn.getHandle(),
                cudnn.scalar(input, 1),
                self.weightBufferDesc[0], self.weightBuffer:data(),
                self.oDesc[0], gradOutput:data(),
                self.convDesc[0],
                bwdDataAlgo,
                extraBuffer, extraBufferSize,
                cudnn.scalar(input, 0),
                self.iDesc[0], self.gradInput:data())
    return self.gradInput
end

function LBORConv:clearDesc()
    self.weightBufferDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.oSize = nil
    self.scaleT = nil
    return self
end

function LBORConv:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function LBORConv:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput', 'input_slice', 'output_slice')
   return nn.Module.clearState(self)
end
