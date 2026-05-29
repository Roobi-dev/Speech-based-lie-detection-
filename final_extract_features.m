function feat = final_extract_features(audio, fs)



%% ==========================
% 3. MFCC
%% ==========================
mfccCoeff = mfcc(audio,fs);

deltaMFCC  = diff(mfccCoeff);
delta2MFCC = diff(deltaMFCC);

deltaMFCC  = [deltaMFCC; zeros(1,size(deltaMFCC,2))];
delta2MFCC = [delta2MFCC; zeros(2,size(delta2MFCC,2))];

%% Statistics
mfccMean = mean(mfccCoeff);
mfccStd  = std(mfccCoeff);

deltaMean = mean(deltaMFCC);
deltaStd  = std(deltaMFCC);

delta2Mean = mean(delta2MFCC);
delta2Std  = std(delta2MFCC);

%% ==========================
% 4. Pitch
%% ==========================
pitchVal = pitch(audio,fs);
pitchMean = mean(pitchVal);
pitchVar  = var(pitchVal);

%% ==========================
% 5. Jitter (Pitch Variation)
%% ==========================
jitter = mean(abs(diff(pitchVal))) / mean(pitchVal);

%% ==========================
% 6. Shimmer (Amplitude Variation)
%% ==========================
ampEnv = abs(audio);
shimmer = mean(abs(diff(ampEnv))) / mean(ampEnv);

%% ==========================
% 7. RMSE
%% ==========================
rmseVal = rms(audio);

%% ==========================
% 8. Spectral Features
%% ==========================
centroid = spectralCentroid(audio,fs);
flatness = spectralFlatness(audio,fs);
rolloff = spectralRolloffPoint(audio,fs);

centMean = mean(centroid); centStd = std(centroid);
flatMean = mean(flatness); flatStd = std(flatness);
rollMean = mean(rolloff); rollStd = std(rolloff);
%% ==========================
% Spectral Entropy
%% ==========================

spectrum = abs(fft(audio));
spectrum = spectrum ./ sum(spectrum + eps);

spectralEntropy = -sum(spectrum .* log2(spectrum + eps));

%% ==========================
% 9. Zero Crossing Rate
%% ==========================
zcr = sum(abs(diff(sign(audio)))) / (2 * length(audio));
zcrMean = mean(zcr);
zcrStd  = std(zcr);

%% ==========================
% FINAL FEATURE VECTOR
%% ==========================
feat = [ ...
    mfccMean mfccStd ...
    deltaMean deltaStd ...
    delta2Mean delta2Std ...
    pitchMean pitchVar ...
    jitter shimmer ...
    rmseVal ...
    centMean centStd ...
    flatMean flatStd ...
    rollMean rollStd ...
    spectralEntropy ...
    zcrMean zcrStd];

end