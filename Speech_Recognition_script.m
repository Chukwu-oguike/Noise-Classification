% To run this script you should download speech data from this link:
% http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
% Extract the files and store them in a folder named speech_data
% Then assign the folder path to the variable data_folder

%data_folder = .../speech_data

ads = audioexample.Datastore(datafolder,'IncludeSubfolders',true,...
                             'FileExtensions','.wav','LabelSource',...
                             'foldernames','ReadMethod','File')

                         
%create the list of words deep network needs to classify
word_Recog = ["left","off","up","right","yes","stop","on","down","no","go","give","take",...
                "stay","run","come","back"];

%tag everything else as unknown
is_Word_Recog = ismember(ads.Labels,categorical(word_Recog));
not_Word_Recog = ~ismember(ads.Labels,categorical([word_Recog,"_background_noise_"]));


%since the number of words outside the l greatly outnumbers the set of words to
%recognize, we take a fraction of the complement set in the training set
prob_Include_Not_Word = 0.1;
mask = rand(numel(ads.Labels),1) < prob_Include_Not_Word;
not_Word_Recog = not_Word_Recog & mask;
ads.Labels(not_Word_Recog) = categorical("not_Word_Recog");


%create datastore data
ads = getSubsetDatastore(ads,is_Word_Recog|not_Word_Recog);


%Partition datastore into training, validation and testing sets
[ads_Train,ads_Validation,ads_Test] = splitData(ads,datafolder);


%Convert speech data in training validation and testing sets to logarithms of mel-spectrograms

%parameters for mel-spectrogram
segment_Duration = 1;
frame_Duration = 0.025;
hop_Duration = 0.010;
num_Bands = 40;
epsilon = 1e-6;


%compute log of mel-spectrograms for training data
data_Training = speechSpectrograms(ads_Train,segment_Duration,frame_Duration,hop_Duration,num_Bands);
data_Training = log10(data_Training + epsilon);

%compute log of mel-spectrograms for validation data
data_Validation = speechSpectrograms(ads_Validation,segment_Duration,frame_Duration,hop_Duration,num_Bands);
data_Validation = log10(data_Validation + epsilon);

%compute log of mel-spectrograms for testing data
data_Testing = speechSpectrograms(ads_Test,segment_Duration,frame_Duration,hop_Duration,num_Bands);
data_Testing = log10(data_Testing + epsilon);

%get data labels
training_data_label = ads_Train.Labels;
validation_data_label = ads_Validation.Labels;
testing_data_label = ads_Test.Labels;


%introduce some noise to training data so network learns to recognize words
%in the presence of noise

ads_noise = copy(ads);

ads_Back_Ground = subset(ads_noise,ads_noise.Labels=="_background_noise_");
num_of_Back_Ground_Clips = 3500;
volume_Range = [1e-4,1];

%calculate log of mel-spectrograms of the clips with background noise
data_w_Bckgrd_Noise = backgroundSpectrograms(ads_Back_Ground,num_of_Back_Ground_Clips,volume_Range,segment_Duration,...
                              frame_Duration,hop_Duration,num_Bands);
data_w_Bckgrd_Noise = log10(data_w_Bckgrd_Noise + epsilon);

%partition these clips with background noise into training, validation and
%testing sets
num_Train_w_Bckgrd_Noise = floor(0.8*num_of_Back_Ground_Clips);
num_Validation_w_Bckgrd_Noise = floor(0.1*num_of_Back_Ground_Clips);
num_Test_w_Bckgrd_Noise = floor(0.1*num_of_Back_Ground_Clips);


%add data with background noise to training data
data_Training(:,:,:,end+1:end+num_Train_w_Bckgrd_Noise) = data_w_Bckgrd_Noise(:,:,:,1:num_Train_w_Bckgrd_Noise);
data_w_Bckgrd_Noise(:,:,:,1:num_Train_w_Bckgrd_Noise) = [];
training_data_label(end+1:end+num_Train_w_Bckgrd_Noise) = "background";

%add data with backgroung noise to validation data
data_Validation(:,:,:,end+1:end+num_Validation_w_Bckgrd_Noise) = data_w_Bckgrd_Noise(:,:,:,1:num_Validation_w_Bckgrd_Noise);
data_w_Bckgrd_Noise(:,:,:,1:num_Validation_w_Bckgrd_Noise) = [];
validation_data_label(end+1:end+num_Validation_w_Bckgrd_Noise) = "background";

%add data with backgroung noise to testing data
data_Testing(:,:,:,end+1:end+num_Test_w_Bckgrd_Noise) = data_w_Bckgrd_Noise(:,:,:,1: num_Test_w_Bckgrd_Noise);
clear data_w_Bckgrd_Noise;
testing_data_label(end+1:end+num_Test_w_Bckgrd_Noise) = "background";

%update labels
training_data_label = removecats(training_data_label);
validation_data_label = removecats(validation_data_label);
testing_data_label = removecats(testing_data_label);


%augment data to reduce the chance on overfitting when training the model
sz = size(data_Training);
specSize = sz(1:2);
image_Size = [specSize 1];
augmenter = imageDataAugmenter('RandXTranslation',[-10 10],'RandXScale',[0.8 1.2],...
                               'FillValue',log10(epsilon));
                           
augimdsTrain = augmentedImageDatastore(image_Size,data_Training,training_data_label, ...
                                       'DataAugmentation',augmenter);

%create convolutional neural network using the function "convo_Net"
conv_net = convo_Net(training_data_label,image_Size);


%set training options
miniBatchSize = 128;
validationFrequency = floor(numel(training_data_label)/miniBatchSize);
options = trainingOptions('adam','InitialLearnRate',3e-4,...
                          'MaxEpochs',25,'MiniBatchSize',miniBatchSize,...
                          'Shuffle','every-epoch','Plots','training-progress',...
                          'Verbose',false,'ValidationData',{data_Validation,validation_data_label},...
                          'ValidationFrequency',validationFrequency, ...
                          'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,...
                          'LearnRateDropPeriod',20);

%train convolutional neural network
trained_conv_net = trainNetwork(augimdsTrain,conv_net,options);


%evaluate model

%evaluate model's accuracy and display results
YValPred = classify(trained_conv_net,data_Validation);
validationError = mean(YValPred ~= validation_data_label);
YTrainPred = classify(trained_conv_net,data_Training);
trainError = mean(YTrainPred ~= training_data_label);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

%plot confusion matrix
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(validation_data_label,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sortClasses(cm, [commands,"unknown","background"])

