local function normalizeGlobal(data, mean_, std_)
   local std = std_ or data:std()
   local mean = mean_ or data:mean()
   data:add(-mean)
   data:mul(1/std)
   return mean, std
end

return function (opt)
   torch.manualSeed(1234)
   local tnt = require 'torchnet'
   local provider = torch.load('./datasets/Outex-24-ori-aug.t7')
   local trainData = {
     data=provider.trainData.data:clone(),
     labels=provider.trainData.labels:clone()
   }
    
   -- split validation set
   local shuffle = torch.randperm(144000)
   provider.trainData = {
      data=torch.DoubleTensor(120000, 3, 32, 32),
      labels=torch.ByteTensor(120000)
    }
   provider.validationData = {
      data=torch.DoubleTensor(24000, 3, 32, 32),
      labels=torch.ByteTensor(24000)
    }
   for i = 1, 120000 do
      local index = shuffle[i]
      provider.trainData.data[i]:copy(trainData.data[index])
      provider.trainData.labels[i]=trainData.labels[index]
   end
   for i = 1, 24000 do
      local index = shuffle[i+120000]
      provider.validationData.data[i]:copy(trainData.data[index])
      provider.validationData.labels[i] = trainData.labels[index]
   end
   trainData = nil
    
   -- preprocess
   local mean 
   local std
   provider.trainData.data = provider.trainData.data:float()
   provider.validationData.data = provider.validationData.data:float()
   provider.testData.data = provider.testData.data:float()
   mean, std = normalizeGlobal(provider.trainData.data, mean, std)
   normalizeGlobal(provider.validationData.data, mean, std)
   normalizeGlobal(provider.testData.data, mean, std)
   collectgarbage()

   -- auto detection
   local sample = provider.testData.data[1]
   opt.numClasses = provider.testData.labels:max()
   opt.imageChannel = sample:size(1)
   opt.imageSize = sample:size(2)
    
   local function rotation(x)
      return image.rotate(x, 2 * math.pi * torch.rand(1)[1], 'bilinear')
   end

   return provider, function (mode)
      return tnt.ParallelDatasetIterator{
         nthread = 1,
         init = function()
            require 'torchnet'
            require 'image'
            require 'nn'
         end,
         closure = function()
            local dataset = provider[mode..'Data']

            local list_dataset = tnt.ListDataset{
               list = torch.range(1, dataset.labels:numel()):long(),
               load = function(idx)
                  return {
                     input = dataset.data[idx]:float(),
                     target = torch.LongTensor{dataset.labels[idx]},
                  }
               end,
            }

            if mode == 'train' then
               if opt.batchSize > 1 then
                  return list_dataset
                     :shuffle()
                     :batch(opt.batchSize, 'skip-last')
               else
                  return list_dataset
                     :shuffle()
               end
            else
               if opt.batchSize > 1 then
                  return list_dataset 
                     :batch(opt.batchSize, 'include-last')
               else 
                  return list_dataset
               end
            end
         end,
      }
   end
end