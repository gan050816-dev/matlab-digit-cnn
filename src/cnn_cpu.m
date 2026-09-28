clear; clc; close all;

%% 1. 加载 MNIST 数据集（每类最多100张）
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

%% 2. 拆分数据集
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomize');
augimdsTrain = augmentedImageDatastore([28 28], imdsTrain);
augimdsTest = augmentedImageDatastore([28 28], imdsTest);

%% 3. CNN网络结构
layers = [
    imageInputLayer([28 28 1])

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

%% 4. 训练参数，强制CPU训练
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsTest, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');  % 关键修改

%% 5. 训练网络
net = trainNetwork(augimdsTrain, layers, options);

%% 6. 测试集准确率计算
YPred = classify(net, augimdsTest);
YTrue = imdsTest.Labels;
acc = sum(YPred == YTrue) / numel(YTrue);
fprintf('测试集准确率: %.2f%%\n', acc*100);

%% 7. 混淆矩阵
figure;
confusionchart(YTrue, YPred);
title('测试集混淆矩阵');

%% 8. 显示部分预测样本
figure;
for i = 1:16
    subplot(4,4,i);
    img = readimage(imdsTest, i);
    label = classify(net, imresize(img, [28 28]));
    imshow(img);
    title(['预测: ', char(label)], 'FontSize', 10);
end
