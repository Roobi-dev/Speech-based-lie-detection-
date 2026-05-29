clc; clear; close all;
rng(0);  % For reproducibility

%% =========================
% SETTINGS
%% =========================
dataFolder = "Dataset/";
targetFs = 16000;
segmentDuration = 2;   % seconds
overlap = 1;           % seconds

files = dir(fullfile(dataFolder, "*.wav"));
numFiles = length(files);

%% =========================
% PREPARE SEGMENTS
%% =========================
disp("Counting total segments for preallocation...");

totalSegments = 0;
for i = 1:numFiles
    [audio, fs] = audioread(fullfile(dataFolder, files(i).name));
    if fs ~= targetFs
        audio = resample(audio,targetFs,fs);
    end
    totalSamples = length(audio);
    segmentLength = segmentDuration * targetFs;
    step = segmentLength - overlap * targetFs;
    numSegs = floor((totalSamples - segmentLength)/step) + 1;
    totalSegments = totalSegments + numSegs;
end

disp("Total segments to extract: " + totalSegments);

%% =========================
% EXTRACT FEATURES
%% =========================
numFeatures = 50;
X = zeros(totalSegments, numFeatures);
Y = strings(totalSegments,1);

segIdx = 1;

disp("Extracting features from audio segments...");
for i = 1:numFiles
    filePath = fullfile(dataFolder, files(i).name);
    [audio, fs] = audioread(filePath);

    if fs ~= targetFs
        audio = resample(audio,targetFs,fs);
        fs = targetFs;
    end
   
    audio = audio(:,1);

%% Pre-emphasis
audio = filter([1 -0.97],1,audio);

%% Noise Reduction
audio = medfilt1(audio,3);
audio = smoothdata(audio,'movmean',5);

%% Bandpass Filter
bpFilt = designfilt('bandpassiir', ...
    'FilterOrder',8, ...
    'HalfPowerFrequency1',80, ...
    'HalfPowerFrequency2',4000, ...
    'SampleRate',fs);

audio = filtfilt(bpFilt,audio);
    segmentLength = segmentDuration * fs;
    step = segmentLength - overlap * fs;
    totalSamples = length(audio);
    startIdx = 1;

    while (startIdx + segmentLength - 1) <= totalSamples
        segment = audio(startIdx : startIdx + segmentLength - 1);
        %% =========================
% VOICE ACTIVITY DETECTION
%% =========================

energy = rms(segment);

% Skip silent/noisy low-energy regions
if energy < 0.01
    startIdx = startIdx + step;
    continue;
end
        feat = final_extract_features(segment, fs);
        feat(isnan(feat)) = 0;

        if length(feat) < numFeatures
            feat(end+1:numFeatures) = 0;
        elseif length(feat) > numFeatures
            feat = feat(1:numFeatures);
        end

        X(segIdx,:) = feat;

        if contains(lower(files(i).name), "truth")
            Y(segIdx) = "truth";
        else
            Y(segIdx) = "lie";
        end

        segIdx = segIdx + 1;
        startIdx = startIdx + step;
    end
end

Y = categorical(Y);
X = X(1:segIdx-1,:);
Y = Y(1:segIdx-1);

%% =========================
% NORMALIZE FEATURES
%% =========================
[X, mu, sigma] = zscore(X);

%% =========================
% 5-FOLD CV WITH FEATURE SELECTION
%% =========================
cv = cvpartition(Y,'KFold',5);
acc = zeros(5,1);

% For confusion matrix
allPred = [];
allTrue = [];

disp("Performing 5-fold CV...");

for k = 1:5
    trainIdx = training(cv,k);
    testIdx  = test(cv,k);

    Xtrain = X(trainIdx,:);
    Ytrain = Y(trainIdx);
    Xtest  = X(testIdx,:);
    Ytest  = Y(testIdx);

    % Feature selection
    tempModel = fitcensemble(Xtrain,Ytrain,...
        'Method','AdaBoostM1',...
        'NumLearningCycles',200,...
        'Learners',templateTree('MaxNumSplits',5));

    importance = predictorImportance(tempModel);
    [~, idx] = sort(importance,'descend');
    topFeatures = idx(1:30);

    % Train final model for this fold
    model = fitcensemble(Xtrain(:,topFeatures), Ytrain, ...
        'Method','AdaBoostM1', ...
        'NumLearningCycles',400, ...
        'Learners',templateTree('MaxNumSplits',8));

    pred = predict(model, Xtest(:,topFeatures));

    acc(k) = mean(pred == Ytest) * 100;
    fprintf("Fold %d Accuracy = %.2f %%\n", k, acc(k));

    % Store for confusion matrix
    allPred = [allPred; pred];
    allTrue = [allTrue; Ytest];

    if k == 5
        finalTopFeatures = topFeatures;
    end
end

fprintf("\nFinal 5-Fold Accuracy: %.2f %%\n", mean(acc));

%% =========================
% CONFUSION MATRIX
%% =========================
figure;
cm = confusionchart(allTrue, allPred);
cm.Title = 'Confusion Matrix (5-Fold CV)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% =========================
% METRICS (Precision, Recall, F1)
%% =========================
confMat = confusionmat(allTrue, allPred);

TP = confMat(1,1);
FN = confMat(1,2);
FP = confMat(2,1);
TN = confMat(2,2);

precision = TP / (TP + FP);
recall    = TP / (TP + FN);
f1        = 2 * (precision * recall) / (precision + recall);

fprintf("\nPrecision: %.2f\n", precision);
fprintf("Recall: %.2f\n", recall);
fprintf("F1 Score: %.2f\n", f1);

%% =========================
% TRAIN FINAL MODEL
%% =========================
finalModel = fitcensemble(X(:,finalTopFeatures), Y, ...
    'Method','AdaBoostM1', ...
    'NumLearningCycles',400, ...
    'Learners',templateTree('MaxNumSplits',8));

disp("Final model trained.");

%% =========================
% SAVE MODEL
%% =========================
save('deceptionSegmentModel.mat','finalModel','mu','sigma','finalTopFeatures');

disp("Model saved successfully.");
