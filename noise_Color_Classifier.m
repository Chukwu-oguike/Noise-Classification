
%create 1000 random samples of white, brown and pink noise respectively

fs = 50000;
duration = 0.5;
N = duration*fs;

white_Noise = 2*rand([N,1000]) - 1;
wLabels = repelem(categorical("white"),1000,1);

brown_Noise = filter(1,[1,-0.999],white_Noise);
brown_Noise = brown_Noise./max(abs(brown_Noise),[],'all');
bLabels = repelem(categorical("brown"),1000,1);

pink_Noise = pinknoise([N,1000]);
pLabels = repelem(categorical("pink"),1000,1);


%partition the samples into training and validation

%700 samples for training
audio_Train = [white_Noise(:,1:700),brown_Noise(:,1:700),pink_Noise(:,1:700)];
labels_Train = [wLabels(1:700);bLabels(1:700);pLabels(1:700)];

%300
audio_Validation = [wNoise(:,701:end),bNoise(:,701:end),pNoise(:,701:end)];
labels_Validation = [wLabels(701:end);bLabels(701:end);pLabels(701:end)];


%extract centroid and slope of the mel frequency spectrum as features
%from training data and validation data

%create an audio feature extract object, specifying features
audio_feature_object = audioFeatureExtractor("SampleRate",fs,"SpectralDescriptorInput",...
                            "melSpectrum","spectralCentroid",true, ...
                            "spectralSlope",true);
 
%use audio feature extract object to extract similar features from training samples                        
features_Train = extract(audio_feature_object,audio_Train);
[num_Hops_Per_Sequence,num_Features,num_Signals] = size(features_Train);

%convert features to cell arrays of sequences for deep network (as the deep
%nets input layer is a sequenceInputLayer)
features_Train = permute(features_Train,[2,1,3]);
features_Train = squeeze(num2cell(features_Train,[1,2]));



%num_Signals = numel(features_Train)
%use audio feature extract object to extract similar features from validation samples
[num_Features,num_Hops_Per_Sequence] = size(features_Train{1})
features_Validation = extract(audio_feature_object,audio_Validation);

%convert features to cell arrays of sequences for deep network (as the deep
%nets input layer is a sequenceInputLayer)
features_Validation = permute(features_Validation,[2,1,3]);
features_Validation = squeeze(num2cell(features_Validation,[1,2]));

%define layers of network
layers = [ ...
          sequenceInputLayer(num_Features)
          lstmLayer(50,"OutputMode","last")
          fullyConnectedLayer(numel(unique(labels_Train)))
          softmaxLayer
          classificationLayer];
 
%define training options      
options = trainingOptions("adam","Shuffle","every-epoch","ValidationData",...
                          {features_Validation,labels_Validation}, ...
                          "Plots","training-progress","Verbose",false);
%train network
deep_net = trainNetwork(features_Train,labels_Train,layers,options);


%test network on validation data
wNoiseTest = 2*rand([N,1]) - 1;
classify(deep_net,extract(audio_feature_object,wNoiseTest)')

bNoiseTest = filter(1,[1,-0.999],wNoiseTest);
bNoiseTest= bNoiseTest./max(abs(bNoiseTest),[],'all');
classify(deep_net,extract(audio_feature_object,bNoiseTest)')

pNoiseTest = pinknoise(N);
classify(deep_net,extract(audio_feature_object,pNoiseTest)')