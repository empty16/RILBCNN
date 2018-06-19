local function normalizeGlobal(data, mean_, std_)
   local std = std_ or data:std()
   local mean = mean_ or data:mean()
   data:add(-mean)
   data:mul(1/std)
   return mean, std
end

--[[
the Entrance function of the lua file
path.dofile is a function to test the code
]] 
return function (opt)
   torch.manualSeed(1234)
   local tnt = require 'torchnet'
   local provider = torch.load('./datasets/MNIST.t7')
   local trainData = {
     data = provider.trainData.data:clone(),
     labels = provider.trainData.labels:clone()
   }
    
   -- split train data to train + validation set
   local shuffle = torch.randperm(60000)
   provider.trainData = {
      data = torch.DoubleTensor(50000, 1, 32, 32),
      labels = torch.ByteTensor(50000)
    }
   provider.validationData = {
      data = torch.DoubleTensor(10000, 1, 32, 32),
      labels = torch.ByteTensor(10000)
    }

    -- train dataset
   for i = 1, 50000 do
      local index = shuffle[i]
      provider.trainData.data[i]:copy(trainData.data[index])
      provider.trainData.labels[i] = trainData.labels[index]
   end

   -- validation dataset
   for i = 1, 10000 do
      local index = shuffle[i+50000]
      provider.validationData.data[i]:copy(trainData.data[index])
      provider.validationData.labels[i] = trainData.labels[index]
   end
   trainData = nil
    
   -- data pre-processing
   local mean
   local std
   provider.trainData.data = provider.trainData.data:float()
   provider.validationData.data = provider.validationData.data:float()
   provider.testData.data = provider.testData.data:float()

   -- using training data to calculate the mean & std and applied to all data
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

   local function original(x)
      return x
   end
   
   --[[
      anonymous function used in lua, python
      multiple return values and functional coding
      the real returned value
      provider, dataIterator = func()
   ]] 

   -- mode = trian / test
   return provider, function (mode)
      --self,init,closure,nthread,filter,transform,ordered
      return tnt.ParallelDatasetIterator{
         nthread = 1,      --key 1
         init = function() --key 2 
            require 'torchnet'
            require 'image'
            require 'nn'
         end,
         closure = function() --key 3 list_data
            -- use the provider[mode..'Data'] to extrcat mode and get the dataset
            local dataset = provider[mode..'Data']

            -- self,list,load,path
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
                     -- input = function
                     :transform{input = original}
                     :batch(opt.batchSize, 'skip-last')
               else
                  return list_dataset
                     :shuffle()
                     :transform{input = original}
               end
            else
               if opt.batchSize > 1 then
                  return list_dataset
                     :transform{input = rotation}
                     :batch(opt.batchSize, 'include-last')
               else 
                  return list_dataset
                        :transform{input = rotation}
               end
            end
         end,
      }
   end
end
