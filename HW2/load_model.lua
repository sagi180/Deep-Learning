require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'gnuplot'

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

	function BatchFlip:updateOutput(input)
		input:float()
		self.output:set(input:cuda())
		return self.output
	end
end

function forwardNet(data, labels)
	batchSize = 128
	criterion = nn.ClassNLLCriterion():cuda()
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    model:evaluate() -- turn off dropout
	
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError
end

function loadModel(model_path, testData, testLabels)
	model = torch.load(model_path)
	testLoss, testError = forwardNet(testData, testLabels)
	return testError
end

testset = torch.load('cifar.torch/cifar10-test.t7')
testData = testset.data:float()
testLabels = testset.label:float():add(1)
-- normalize with the mean and std of the training set
mean = {125.30691804687, 122.95039414062, 113.86538318359}  -- mean values (per channel)
stdv  = {62.993219892912, 62.088708246722, 66.704900292063} -- standard deviation values (per channel)
-- normalize each channel
for i=1,3 do
	testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
	testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

testError = loadModel('best_model.dat', testData, testLabels);
print('*** Test error: ' .. testError)