require 'nn'
require 'cunn'
require 'optim'
require 'gnuplot'

mnist = require 'mnist';

-- ### Parameters ###

function initParams()
	inputSize = 28*28
	layerSize = {inputSize, 64,32,64,128}

	batchSize = 256

	epochs = 1000

	optimState = {
		learningRate = 0.001,
		beta1 = 0.9,
		beta2 = 0.999,
		epsilon = 1e-8,
	}
	optimFunc = optim.adam
end

initParams();
	
-- ### End of Parameters ###

trainData = mnist.traindataset().data:float();
trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

-- We'll start by normalizing our data
mean = trainData:mean()
std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);

----- ### Shuffling data

function shuffle(data, labels) --shuffle data function
    randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

function sample(from,to,size)
	local indexes = torch.Tensor(size)
	for i = 1, size do
		indexes[i] = torch.random(from, to)
	end
	return indexes
end

function shuffleWithReturns(data, labels) --shuffle data function with returns
    randomIndexes = sample(1, data:size(1), data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

--- ### Main evaluation + training function

function forwardNet(data, labels, train, optimFunc)
    timer = torch.Timer()
    --another helpful function of optim is ConfusionMatrix
    confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    lossAcc = 0
    numBatches = 0
    if train then
        --set network into training mode
        model:training()
	else
        --set network into test mode
		model:evaluate()
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        x = data:narrow(1, i, batchSize):cuda()
        yt = labels:narrow(1, i, batchSize):cuda()
        y = model:forward(x)
        err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
                return err, dE_dw
            end
            optimFunc(feval, w, optimState)
        end
    end
    confusion:updateValids()
    avgLoss = lossAcc / numBatches
    avgError = 1 - confusion.totalValid
    return avgLoss, avgError, tostring(confusion)
end

--- ### Train the network on training set, evaluate on separate set

function trainAndEval(model, trainData, trainLabels, testData, testLabels, epochs, optimFunc)
	trainLoss = torch.Tensor(epochs)
	testLoss = torch.Tensor(epochs)
	trainError = torch.Tensor(epochs)
	testError = torch.Tensor(epochs)
	maxTestAcc = 0
	maxTestAccEpoch = 0

	-- Reset net weights
	model:apply(function(l) l:reset() end)

	for e = 1, epochs do
		trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
		trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true, optimFunc)
		testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false, optimFunc)
		testAcc = (1 - testError[e]) * 100
		--[[if e % 10 == 0 then
			print('Epoch ' .. e .. ':')
			print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
			print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
			print('Test accuracy: ' .. testAcc)
		end]]--
		if maxTestAcc < testAcc then
			maxTestAcc = testAcc
			maxTestAccEpoch = e
		end
	end

	print('*** Max test accuracy: ' .. maxTestAcc, 'epoch ' .. maxTestAccEpoch)

	-- Plot
	range = torch.range(1, epochs)
	gnuplot.pngfigure('error_graph.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
	range = torch.range(1, epochs)
	gnuplot.pngfigure('loss_graph.png')
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()
	
	return maxTestAccEpoch
end

function createModel()
	model = nn.Sequential()
	model:add(nn.View(inputSize)) --reshapes the image into a vector without copy
	model:add(nn.Dropout(0.4):cuda())
	model:add(nn.Linear(inputSize, 64))
	model:add(nn.RReLU())
	model:add(nn.Linear(64, 32))
	model:add(nn.RReLU())
	model:add(nn.Linear(32, 64))
	model:add(nn.RReLU())
	model:add(nn.Linear(64, 128))
	model:add(nn.RReLU())
	model:add(nn.Dropout(0.5):cuda())
	model:add(nn.Linear(128, outputSize))
	model:add(nn.LogSoftMax())

	model:cuda() --ship to gpu
	return model
end

------   ### Define model and criterion

outputSize = 10

model = createModel()
print(tostring(model))

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())

---- ### Classification criterion
criterion = nn.ClassNLLCriterion():cuda()

epochs = trainAndEval(model, trainData, trainLabels, testData, testLabels, epochs, optimFunc)

-- Save the model
torch.save('model.dat', model)
