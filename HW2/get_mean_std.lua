require 'torch'
trainset = torch.load('cifar.torch/cifar10-train.t7')
testset = torch.load('cifar.torch/cifar10-test.t7')

trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
trainLabels = trainset.label:float():add(1)
testData = testset.data:float()
testLabels = testset.label:float():add(1)

-- Load and normalize data

mean = {}  -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
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

print(mean)
print(stdv)