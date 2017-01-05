require 'nn'
require 'cunn'
require 'optim'
mnist = require 'mnist';

function forwardNet(data, labels)
	batchSize = 256
	criterion = nn.ClassNLLCriterion():cuda()
    confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    lossAcc = 0
    numBatches = 0
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        x = data:narrow(1, i, batchSize):cuda()
        yt = labels:narrow(1, i, batchSize):cuda()
        y = model:forward(x)
        err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end
    confusion:updateValids()
    avgLoss = lossAcc / numBatches
    avgError = 1 - confusion.totalValid
    return avgLoss, avgError
end

function loadModel(model_path, testData, testLabels)
	model = torch.load(model_path)
	testLoss, testError, confusion = forwardNet(testData, testLabels)
	return testError
end

testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);
-- normalize with the mean and standard deviation of the training set
mean = 33.31842144983;
std = 78.567490830614;
testData:add(-mean):div(std);

testError = loadModel('model_TEST.dat', testData, testLabels);
print('*** Test error: ' .. testError)