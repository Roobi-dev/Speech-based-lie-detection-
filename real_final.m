clc; clear; close all;
rng(0);  % reproducibility

%% =========================
% Load Model
%% =========================
load('deceptionSegmentModel.mat'); % finalModel, mu, sigma, finalTopFeatures

%% =========================
% Select Audio File
%% =========================
[file,path] = uigetfile('*.wav','Select Audio File');
if isequal(file,0)
    error('No file selected.');
end
filePath = fullfile(path,file);

[audio, fs] = audioread(filePath);
targetFs = 16000;

if fs ~= targetFs
    audio = resample(audio,targetFs,fs);
    fs = targetFs;
end

audio = audio(:,1);  % mono
audio = filter([1 -0.97],1,audio);  % pre-emphasis

%% =========================
% Segment Parameters
%% =========================
segmentDuration = 2;          % seconds
overlap = 1;                  % seconds (50% overlap)
segmentLength = segmentDuration * fs;
step = segmentLength - overlap * fs;

numSegments = ceil((length(audio) - segmentLength)/step) + 1;

%% =========================
% Preallocate Feature Matrix
%% =========================
numFeatures = length(finalTopFeatures);
Xtest = zeros(numSegments, numFeatures);

%% =========================
% Extract Features per Segment
%% =========================
for s = 1:numSegments
    startIdx = (s-1)*step + 1;
    endIdx = startIdx + segmentLength - 1;

    % Handle last short segment
    if endIdx > length(audio)
        segment = audio(startIdx:end);
        segment(end+1:segmentLength) = 0; % zero-padding
    else
        segment = audio(startIdx:endIdx);
    end

    feat = final_extract_features(segment, fs);
    feat(isnan(feat)) = 0;

    % Ensure feat has enough length
    if length(feat) < max(finalTopFeatures)
        feat(end+1:max(finalTopFeatures)) = 0;
    end

    Xtest(s,:) = feat(finalTopFeatures);
end

%% =========================
% Normalize Features
%% =========================
Xtest = (Xtest - mu(finalTopFeatures)) ./ sigma(finalTopFeatures);

%% =========================
% Predict Segment Scores
%% =========================
[~, score] = predict(finalModel, Xtest);

truthIndex = find(strcmp(cellstr(finalModel.ClassNames), 'truth'));
lieIndex   = find(strcmp(cellstr(finalModel.ClassNames), 'lie'));

rawTruth = score(:, truthIndex);
rawLie   = score(:, lieIndex);

% Convert raw AdaBoost scores to probabilities using logistic scaling
probTruth = 1 ./ (1 + exp(-rawTruth));
probLie   = 1 ./ (1 + exp(-rawLie));

% Average segment probabilities
truthPercentage = mean(probTruth) * 100;
liePercentage   = mean(probLie) * 100;

%% =========================
% Final Decision
%% =========================
if truthPercentage >= liePercentage
    finalDecision = 'TRUTH';
else
    finalDecision = 'LIE';
end

%% =========================
% Display Results
%% =========================
fprintf('\n=============================\n');
fprintf('Final Decision: %s\n', finalDecision);
fprintf('Truth Confidence: %.2f %%\n', truthPercentage);
fprintf('Lie Confidence:   %.2f %%\n', liePercentage);
fprintf('Number of Segments: %d\n', numSegments);
fprintf('=============================\n');