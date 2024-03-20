function [c_net] = convo_Net(y_Train,image_size)

class_Weights = 1./countcats(y_Train);
class_Weights = class_Weights'/mean(class_Weights);
num_Classes = numel(categories(y_Train));

time_Pool_Size = ceil(imageSize(2)/8);
dropout_Prob = 0.2;
num_F = 12;

layers = [
           imageInputLayer(image_size)

           convolution2dLayer(3,num_F,'Padding','same')
           batchNormalizationLayer
           reluLayer

           maxPooling2dLayer(3,'Stride',2,'Padding','same')

           convolution2dLayer(3,2*num_F,'Padding','same')
           batchNormalizationLayer
           reluLayer

           maxPooling2dLayer(3,'Stride',2,'Padding','same')

           convolution2dLayer(3,4*num_F,'Padding','same')
           batchNormalizationLayer
           reluLayer

           maxPooling2dLayer(3,'Stride',2,'Padding','same')

           convolution2dLayer(3,4*num_F,'Padding','same')
           batchNormalizationLayer
           reluLayer
           convolution2dLayer(3,4*num_F,'Padding','same')
           batchNormalizationLayer
           reluLayer

           maxPooling2dLayer([1 time_Pool_Size])

           dropoutLayer(dropout_Prob)
           fullyConnectedLayer(num_Classes)
           softmaxLayer
           weightedClassificationLayer(class_Weights)];

 c_net = layers;
end

