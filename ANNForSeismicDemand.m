% Main script for ANN vs XGBoost comparison
clear; clc; close all;

%% 1. Load data
load('DataForApplCheck.mat'); % Assume X (N×31), Y (N×6)
X=DataForApplCheck.input;
Y=log(DataForApplCheck.output);
[N, nFeat] = size(X);
fprintf('Dataset loaded: %d samples, %d features.\n', N, nFeat);

%% 2. Train/Test split
rng(42); % reproducibility
cv = cvpartition(N, 'HoldOut', 0.2);
Xtrain = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest  = X(test(cv), :);
Ytest  = Y(test(cv), :);

%% 3. Normalization
[xtrain, xPS] = mapminmax(Xtrain', 0, 1); xtrain = xtrain';
[xtest, ~]    = mapminmax('apply', Xtest', xPS); xtest = xtest';

[ytrain, yPS] = mapminmax(Ytrain', 0, 1); ytrain = ytrain';
[ytest, ~]    = mapminmax('apply', Ytest', yPS); ytest = ytest';

%% 4. Train ANN model
fprintf('--- Training ANN ---\n');
tic;
annModel = trainANN(xtrain, ytrain);
annTrainTime = toc;

%% ANN inference
tic;
Ypred_ann = annModel(xtest')'; % output normalized
annInferTime = toc / size(xtest,1);

% Reverse normalization
Ypred_ann_real = mapminmax('reverse', Ypred_ann', yPS)'; 

%% 5. Train ANN with automatic hidden-neuron search + K-fold CV
fprintf('--- Training ANN with CV Search ---\n');
tic;
[annModel, bestH] = trainANN(xtrain, ytrain);
annTrainTime = toc;

tic;
Ypred_ann = annModel(xtest')'; % normalized
annInferTime = toc / size(xtest,1);
Ypred_ann_real = mapminmax('reverse', Ypred_ann', yPS)'; 


%% XGBoost inference
%% ================== XGBoost 训练 ==================
tic;
xgbModel = trainXGB(xtrain, ytrain);
xgbTrainTime = toc;

tic;
Ypred_xgb = zeros(size(ytest));
for j = 1:size(Ytest,2)
    Ypred_xgb(:,j) = predict(xgbModel{j}, xtest);
end
xgbInferTime = toc/size(xtest,1);

% 反标准化
Ypred_xgb_real = mapminmax('reverse',Ypred_xgb', yPS)'; 
%% 6. Performance comparison
mse_ann = mean((Ytest - Ypred_ann_real).^2, 'all');
r2_ann  = 1 - sum((Ytest - Ypred_ann_real).^2, 'all')/sum((Ytest - mean(Ytest)).^2, 'all');

mse_xgb = mean((Ytest - Ypred_xgb_real).^2, 'all');
r2_xgb  = 1 - sum((Ytest - Ypred_xgb_real).^2, 'all')/sum((Ytest - mean(Ytest)).^2, 'all');

fprintf('\n--- Runtime & Accuracy Comparison ---\n');
fprintf('Model     | Train Time (s) | Inference Time (s/sample) | MSE      | R^2\n');
fprintf('ANN       | %.3f           | %.6f                     | %.4f   | %.4f\n', ...
        annTrainTime, annInferTime, mse_ann, r2_ann);
fprintf('XGBoost   | %.3f           | %.6f                     | %.4f   | %.4f\n', ...
        xgbTrainTime, xgbInferTime, mse_xgb, r2_xgb);

%% Plot comparison for one output (e.g., first seismic demand)
%% 结果展示：2x3 子图散点对比六个输出
figure;
for i = 1:6
    subplot(2,3,i);
    scatter(exp(Ytest(:,i)), exp(Ypred_ann_real(:,i)), 40, 'r', 'filled','DisplayName','ANN Pred'); hold on;
    scatter(exp(Ytest(:,i)), exp(Ypred_xgb_real(:,i)), 40, 'b', 'filled','DisplayName','XGBoost Pred');
    plot([min(exp(Ytest(:,i))), max(exp(Ytest(:,i)))], [min(exp(Ytest(:,i))), max(exp(Ytest(:,i)))], 'k--','LineWidth',1.5,'DisplayName','y=x');
    xlabel('True Value');
    ylabel(sprintf('Predicted Output %d',i));
    title(sprintf('Scatter Comparison - Output %d',i));
    legend('Location','best');
    grid on;
end
sgtitle('Scatter Plots: ANN vs XGBoost Prediction Results (All 6 Outputs)');


%% --- Functions ---

function [bestModel, bestH] = trainANN(X, Y)
    % trainANN - Search best hidden neurons (20~100) with K-fold CV
    % Input:  X (N×features), Y (N×outputs)
    % Output: bestModel (trained ANN), bestH (best hidden neurons)
    
    hiddenCandidates = 20:2:100;
    K = 5; % K-fold
    cvPart = cvpartition(size(X,1), 'KFold', K);

    bestMSE = inf;
    bestH = 0;
    meanMSEs = zeros(length(hiddenCandidates),1);

    for i = 1:length(hiddenCandidates)
        h = hiddenCandidates(i);
        mseFolds = zeros(K,1);

        for k = 1:K
            idxTrain = training(cvPart,k);
            idxVal   = test(cvPart,k);

            % Create ANN
            net = feedforwardnet(h, 'trainlm');
            net.trainParam.epochs = 10;
            net.trainParam.goal = 1e-3;
            net.divideParam.trainRatio = 1.0;
            net.divideParam.valRatio   = 0.0;
            net.divideParam.testRatio  = 0.0;

            % Train on training fold
            net = train(net, X(idxTrain,:)', Y(idxTrain,:)');
            Yval_pred = net(X(idxVal,:)')';
            
            % Fold MSE
            mseFolds(k) = mean((Y(idxVal,:) - Yval_pred).^2, 'all');
        end

        meanMSE = mean(mseFolds);
        meanMSEs(i) = meanMSE;

        fprintf('Hidden %d -> CV-MSE = %.4f\n', h, meanMSE);

        if meanMSE < bestMSE
            bestMSE = meanMSE;
            bestH = h;
        end
    end

    fprintf('Best hidden neurons = %d (CV-MSE=%.4f)\n', bestH, bestMSE);

    % Retrain best model on all data
    bestModel = feedforwardnet(bestH, 'trainlm');
    bestModel.trainParam.epochs = net.trainParam.epochs;
    bestModel.trainParam.goal = 1e-3;
    bestModel = train(bestModel, X', Y');

    % ---- 绘制隐含层数 vs CV-MSE 曲线 ----
    figure;
    plot(hiddenCandidates, meanMSEs, '-o','LineWidth',1.5);
    xlabel('Hidden neurons');
    ylabel('Mean CV-MSE');
    title('K-fold CV performance for different hidden neurons');
    grid on;
end


function Ypred = predictXGBmulti(model, Xtest)
    nOut = numel(model.submodels);
    N = size(Xtest,1);
    Ypred = zeros(N,nOut);
    for j = 1:nOut
        Ypred(:,j) = predict(model.submodels{j}, Xtest);
    end
end

function model = trainXGB(Xtrain, Ytrain)
    % trainXGB - 使用梯度提升树 (作为XGBoost替代) 训练回归模型
    % Input:  Xtrain (N×features), Ytrain (N×outputs)
    % Output: model (cell array，每个输出一个回归模型)

    nOutputs = size(Ytrain,2);
    model = cell(1, nOutputs);

    for j = 1:nOutputs
        % 每个输出单独建模
        model{j} = fitrensemble(Xtrain, Ytrain(:,j), ...
            'Method','LSBoost', ...   % 梯度提升
            'NumLearningCycles', 300, ... % 树数
            'Learners', templateTree('MaxNumSplits', 30), ...
            'LearnRate', 0.1);
    end
end



