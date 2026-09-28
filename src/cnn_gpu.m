clear; clc; close all;

digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', ...
    'nndatasets', 'DigitDataset');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

minNum = 100;
labels = unique(imds.Labels);
selectedFiles = [];

for i = 1:numel(labels)
    idx = find(imds.Labels == labels(i));
    idx = idx(1:min(minNum, numel(idx)));
    selectedFiles = [selectedFiles; imds.Files(idx)];
end

imds = imageDatastore(selectedFiles, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomize');

% 网络输入是 28x28x3
inputSize = [28 28 3];

% 用gray2rgb自动把灰度图变成3通道RGB
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 64, ...      % 改小batch，防止显存不足
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsTest, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');  % 指定用GPU训练

gpuDevice([]); % 重置GPU

net = trainNetwork(augimdsTrain, layers, options);

% 分类时也指定GPU，且设置MiniBatchSize
YPred = classify(net, augimdsTest, 'ExecutionEnvironment', 'gpu', 'MiniBatchSize', 64);

YTrue = imdsTest.Labels;
acc = sum(YPred == YTrue) / numel(YTrue);
fprintf('测试集准确率: %.2f%%\n', acc*100);

figure;
confusionchart(YTrue, YPred);
title('测试集混淆矩阵');

figure;
for i = 1:16
    subplot(4,4,i);
    img = readimage(imdsTest, i);
    img_rgb = repmat(img, [1 1 3]); % 确保3通道
    img_resized = imresize(img_rgb, inputSize(1:2));
    label = classify(net, img_resized, 'ExecutionEnvironment', 'gpu');
    imshow(img);
    title(['预测: ', char(label)], 'FontSize', 10);
end
