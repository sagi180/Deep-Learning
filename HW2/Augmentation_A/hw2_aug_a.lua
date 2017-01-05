require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'gnuplot'
local colorize = require 'trepl.colorize'
-- Class the define the parameters

ConfigData = {}
ConfigData.__index = ConfigData
function ConfigData.create(name,epoch,batchSize,a,b,c,d,e,f,drop1,drop2)
   local acnt = {}             -- our new object
   setmetatable(acnt,ConfigData)  -- make ConfigData handle lookup
   acnt.name = name      -- initialize our object
   acnt.epoch = epoch      -- initialize our object
   acnt.batchSize = batchSize      -- initialize our object
   acnt.a = a      -- initialize our object
   acnt.b = b      -- initialize our object
   acnt.c = c      -- initialize our object
   acnt.d = d      -- initialize our object
   acnt.e = e      -- initialize our object
   acnt.f = f      -- initialize our object
   acnt.drop1 = drop1      -- initialize our object
   acnt.drop2 = drop2      -- initialize our object
   return acnt
end

function getCategory()
	c = torch.random(1,100)
	if (c <= c1) then return 1
	end
	if (c > c1 and c <= c2) then return 2
	end
	if (c > c2 and c <= c3) then return 3
	end
	if (c > c3 and c <= c4) then return 4
	end
	if (c > c4 and c <= c5) then return 5
	end
	if (c > c5) then return 6
	end
end

function imgAugmentation(img)
	rndCategory = getCategory()
	imageRef = require 'image'
	if rndCategory == 1 then
		return imageRef.hflip(img)
	end
	if rndCategory == 2 then
		theta = torch.random(5,12)
		m = nn.SpatialReflectionPadding(5,5,5,5):float()
		padRotate = m:forward(img:float())
		padRotate = imageRef.rotate(padRotate,theta*0.0174532925)
		return padRotate:narrow(3,6,32):narrow(2,6,32)
	end
	if rndCategory == 3 then
		theta = torch.random(5,12)
		m = nn.SpatialReflectionPadding(5,5,5,5):float()
		padRotate = m:forward(img:float())
		padRotate = imageRef.rotate(padRotate,-theta*0.0174532925)
		return padRotate:narrow(3,6,32):narrow(2,6,32)
	end
	if rndCategory == 4 then
		theta = torch.random(5,12)
		m = nn.SpatialReflectionPadding(5,5,5,5):float()
		padRotate = m:forward(imageRef.hflip(img):float())
		padRotate = imageRef.rotate(padRotate,theta*0.0174532925)
		return padRotate:narrow(3,6,32):narrow(2,6,32)
	end
	if rndCategory == 5 then
		theta = torch.random(5,12)
		m = nn.SpatialReflectionPadding(5,5,5,5):float()
		padRotate = m:forward(imageRef.hflip(img):float())
		padRotate = imageRef.rotate(padRotate,-theta*0.0174532925)
		return padRotate:narrow(3,6,32):narrow(2,6,32)
	else		--rndCategory == 5
		return img
	end
end

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init(a,b,c,d,e,f)
    parent.__init(self,a,b,c,d,e,f)
    self.train = true
	n=a+b+c+d+e+f
	p_hflip = a/n						-- Category 1
	p_randomcrop_reflection = b/n		-- Category 2
	p_randomcrop_zero = c/n				-- Category 3
	p_minmax = d/n						-- Category 4
	p_zoomout = e/n						-- Category 5
	p_non = f/n							-- Category 6 (the rest)

	c1 = 100 * p_hflip
	c2 = 100 * (p_hflip + p_randomcrop_reflection) 
	c3 = 100 * (p_hflip + p_randomcrop_reflection + p_randomcrop_zero)
	c4 = 100 * (p_hflip + p_randomcrop_reflection + p_randomcrop_zero + p_minmax)
	c5 = 100 * (p_hflip + p_randomcrop_reflection + p_randomcrop_zero + p_minmax + p_zoomout)
	--c6 = 100 * (p_hflip + p_randomcrop_reflection + p_randomcrop_zero + p_minmax + p_zoomout + p_non)
  end

	function BatchFlip:updateOutput(input)
		input:float()
		if self.train then
			for i=1,input:size(1) do
			 input[i] = imgAugmentation(input[i]):float()
			end
		end
		self.output:set(input:cuda())
		return self.output
	end
end

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end
--  
--  Training the network
--  
function forwardNet(data,labels, train, optimFunc , batchSize , model , optimState)
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
            optimFunc(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError
end

function plot(train, test, label, configName)
	local range = torch.range(1, train:size(1))
	gnuplot.pngfigure('graph_' .. label .. '_' .. configName .. '.png')
	gnuplot.plot({'train', train},{'test', test})
	gnuplot.xlabel('Epochs')
	gnuplot.ylabel(label)
	gnuplot.plotflush()
end

--- ### Train the network on training set, evaluate on test set
function trainAndEval(model, trainData, trainLabels, testData, testLabels, epochs, optimFunc, batchSize , optimState, configName)
	trainLoss = torch.Tensor(epochs)
	testLoss = torch.Tensor(epochs)
	trainError = torch.Tensor(epochs)
	testError = torch.Tensor(epochs)
	maxTestAcc = 0
	maxTestAccEpoch = 0

	-- Reset net weights
	model:apply(function(l) l:reset() end)
	
	print(os.date("%H:%M:%S") .. ': training... ')

	for e = 1, epochs do
		trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
		trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true, optimFunc , batchSize , model , optimState)
		testLoss[e], testError[e] = forwardNet(testData, testLabels, false, optimFunc , batchSize , model , optimState)
		trainAcc = (1 - trainError[e]) * 100
		testAcc = (1 - testError[e]) * 100
		
		if maxTestAcc < testAcc then
			maxTestAcc = testAcc
			maxTestAccEpoch = e
			-- Save the model
			torch.save('model_' .. configName .. '.dat', model)
		end
		
		print(os.date("%H:%M:%S") .. ' epoch ' .. e .. ': test accuracy ' .. testAcc, 'train accuracy ' .. trainAcc)

	end

	print('*** ' .. os.date("%H:%M:%S") .. ' max test accuracy: ' .. maxTestAcc, ', in epoch ' .. maxTestAccEpoch)

	plot(trainError, testError, 'Error', configName)
	plot(trainLoss, testLoss, 'Loss', configName)

	return maxTestAccEpoch
end
	
function createModel(a,b,c,d,e,f,drop1,drop2)
	model = nn.Sequential()
	
	model:add(nn.BatchFlip(a,b,c,d,e,f))

	model:add(cudnn.SpatialConvolution(3, 32, 3, 3, 1,1, 1,1)) -- 3 input image channel, 32 output channels, 3x3 convolution kernel
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
	model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
	
	model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1,1, 1,1))
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(64))
	model:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
	model:add(nn.Dropout(drop1))
	
	model:add(cudnn.SpatialConvolution(64, 32, 3, 3, 1,1, 1,1))
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
	model:add(cudnn.SpatialMaxPooling(2,2,2,2))
		
	model:add(cudnn.SpatialConvolution(32, 32, 2, 2, 1,1, 1,1))
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(32))
	
	model:add(cudnn.SpatialConvolution(32, 32, 2, 2, 1,1, 1,1))
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(32))
	model:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
	model:add(nn.Dropout(drop2))
	
	model:add(cudnn.SpatialConvolution(32, 10, 2, 2, 1,1, 1,1))
	model:add(cudnn.ReLU(true))
	model:add(nn.SpatialBatchNormalization(10))
	model:add(cudnn.SpatialMaxPooling(2,2,2,2))
	model:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
	
	model:add(nn.View(#classes))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
	model:add(nn.LogSoftMax())  
	model:cuda()
	return model
end
	
function runModel(configuration)
	batchSize = configuration.batchSize
	epoch = configuration.epoch
	a = configuration.a
	b = configuration.b
	c = configuration.c
	d = configuration.d
	e = configuration.e
	f = configuration.f
	drop1 = configuration.drop1
	drop2 = configuration.drop2
	
	optimState = {
		learningRate = 0.001,
		beta1 = 0.9,
		beta2 = 0.999,
		weightDecay = 0.0001,
		momentum = 0.9,
	}
	optimFunc = optim.adam
	
	-- ### End of Parameters ###
	
	local trainset = torch.load('cifar.torch/cifar10-train.t7')
	local testset = torch.load('cifar.torch/cifar10-test.t7')

	classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

	local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
	local trainLabels = trainset.label:float():add(1)
	local testData = testset.data:float()
	local testLabels = testset.label:float():add(1)

	-- Load and normalize data

	local mean = {}  -- store the mean, to normalize the test set in the future
	local stdv  = {} -- store the standard-deviation for the future
	for i=1,3 do -- over each image channel
		mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
		--print('Channel ' .. i .. ', Mean ' .. mean[i])
		trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
		
		stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
		--print('Channel ' .. i .. ', Standard Deviation ' .. stdv[i])
		trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
	end
	-- Normalize test set using same values
	for i=1,3 do -- over each image channel
		testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
		testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
	end

	print(colorize.blue '==>' ..' ###########################  configuring model ##################')
	print(configuration.name)
	model = createModel(a,b,c,d,e,f,drop1,drop2)
	print(colorize.blue '==> ' .. tostring(model))
	w, dE_dw = model:getParameters()
	print('Number of parameters:', w:nElement())
	---- ### Classification criterion
	criterion = nn.ClassNLLCriterion():cuda()
	epochs = trainAndEval(model, trainData, trainLabels, testData, testLabels, epoch, optimFunc , batchSize , optimState, configuration.name)

end

batchSize = 128
epoch = 1000
a = 1
b = 1
c = 1
d = 0
e = 0
f =  2
drop1 = 0.3
drop2 = 0.3
name = 'aug_a_batch' .. batchSize .. '_drop1' .. drop1 .. '_drop2' .. drop2 .. '_a' .. a .. '_b' .. b .. '_c' .. c .. '_d' .. d .. '_e' .. e .. '_f' .. f
config = ConfigData.create(name,epoch,batchSize,a,b,c,d,e,f,drop1,drop2)

runModel(config)
