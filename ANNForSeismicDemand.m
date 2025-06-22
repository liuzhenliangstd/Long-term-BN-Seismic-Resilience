%% train_ANN_BridgeSeismicDemand.m
%  Data-driven seismic demand model for bridge networks
clear; clc; close all;

%% 0. Parameters
K               = 10;                 % K-fold CV
testRatio       = 0.20;               % 20 % hold-out for final test
hidLayerRange   = 1:4;                % no. of hidden layers
neuronsRange    = 20:100;          % neurons per hidden layer
learnRateRange  = [1e-3 3e-3 1e-2];   % learning rates
l2Range         = [0 1e-4 1e-3];      % L2 regularisation factors
maxEpochs       = 1e4;
patience        = 200;                % early-stopping patience

%% 1. Load data
load MLtestdata.mat   %# contains inputfeature (N×31) and outputproperty (N×6)
X = inputfeature;     Y = outputproperty;
N = size(X,1);

% shuffle data
rng default
idx = randperm(N);
X = X(idx,:); Y = Y(idx,:);

%% 2. Train/test split
Ntest        = round(testRatio*N);
Xtest        = X(1:Ntest,:);   Ytest = Y(1:Ntest,:);
XtrainAll    = X(Ntest+1:end,:); YtrainAll = Y(Ntest+1:end,:);

%% 3. Grid-search K-fold CV
bestMSE  = inf;   bestNet = [];
bestInfo = struct();

cv = cvpartition(size(XtrainAll,1),'KFold',K);

for nHidden = hidLayerRange
    for nNeuron = neuronsRange
        for lr = learnRateRange
            for l2 = l2Range
                
                mseFold = zeros(K,1);
                
                for k = 1:K
                    trainIdx = cv.training(k);  valIdx = cv.test(k);
                    Xt = XtrainAll(trainIdx,:);    Yt = YtrainAll(trainIdx,:);
                    Xv = XtrainAll(valIdx,:);      Yv = YtrainAll(valIdx,:);
                    
                    % Build network
                    layers = [ ...
                        featureInputLayer(size(X,2),'Normalization','zscore')
                        fullyConnectedLayer(nNeuron,'L2Factor',l2) ...
                        reluLayer];
                    for h = 2:nHidden
                        layers = [layers ...
                            fullyConnectedLayer(nNeuron,'L2Factor',l2) ...
                            reluLayer]; %#ok<AGROW>
                    end
                    layers = [layers ...
                        fullyConnectedLayer(size(Y,2)) ...
                        regressionLayer];
                    
                    options = trainingOptions('adam', ...
                        'InitialLearnRate',lr, ...
                        'MaxEpochs',maxEpochs, ...
                        'MiniBatchSize',128, ...
                        'Shuffle','every-epoch', ...
                        'Verbose',false, ...
                        'Plots','none', ...
                        'ValidationData',{Xv,Yv}, ...
                        'ValidationPatience',patience);
                    
                    net = trainNetwork(Xt,Yt,layers,options);
                    Ypred = predict(net,Xv);
                    mseFold(k) = mean((Ypred - Yv).^2,'all');
                end
                
                cvMSE = mean(mseFold);
                if cvMSE < bestMSE
                    bestMSE  = cvMSE;
                    bestNet  = net;
                    bestInfo = struct('nHidden',nHidden,'nNeuron',nNeuron, ...
                                      'lr',lr,'l2',l2,'cvMSE',cvMSE);
                end
            end
        end
    end
end

fprintf('Best architecture: %d hidden layer(s) × %d neurons, LR=%.4f, L2=%.0e (CV-MSE=%.4f)\n', ...
         bestInfo.nHidden,bestInfo.nNeuron,bestInfo.lr,bestInfo.l2,bestInfo.cvMSE);

%% 4. Retrain on full training set with best hyper-params
% Build final net with best hyper-parameters
layers = [ ...
    featureInputLayer(size(X,2),'Normalization','zscore')
    fullyConnectedLayer(bestInfo.nNeuron,'L2Factor',bestInfo.l2) reluLayer];

for h = 2:bestInfo.nHidden
    layers = [layers ...
        fullyConnectedLayer(bestInfo.nNeuron,'L2Factor',bestInfo.l2) reluLayer];
end

layers = [layers fullyConnectedLayer(size(Y,2)) regressionLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',bestInfo.lr, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',128, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);

bestNet = trainNetwork(XtrainAll,YtrainAll,layers,options);

%% 5. Test performance
YtrainPred = predict(bestNet,XtrainAll);
YtestPred  = predict(bestNet,Xtest);

trainMSE = mean((YtrainPred-YtrainAll).^2,'all');
testMSE  = mean((YtestPred - Ytest).^2,'all');
trainR2  = 1 - sum((YtrainPred-YtrainAll).^2,'all') / sum((YtrainAll-mean(YtrainAll,'all')).^2,'all');
testR2   = 1 - sum((YtestPred - Ytest).^2,'all') / sum((Ytest-mean(Ytest,'all')).^2,'all');

fprintf('Train MSE: %.4f, R2: %.3f | Test MSE: %.4f, R2: %.3f\n', ...
        trainMSE,trainR2,testMSE,testR2);

%% 6.1  Predicted-vs-Actual plots for all outputs (3×2 subplots)
outputNames = {'y1  CurvDuct','y2  BearDisp','y3  AbutDisp', ...
               'y4  GirderDrift','y5  DeckAcc','y6  FoundMoment'};

figure('Name','Predicted vs Actual (Train & Test)','Position',[100 100 1200 800]);

for k = 1:6
    subplot(3,2,k); hold on; box on; grid on;
    
    % Train and test scatter
    scatter(YtrainAll(:,k),YtrainPred(:,k),10,'b','filled','MarkerFaceAlpha',0.4);
    scatter(Ytest(:,k),    YtestPred(:,k), 10,'r','filled','MarkerFaceAlpha',0.6);
    
    % 1:1 reference line
    ax = gca;
    lims = [min(ax.XLim(1),ax.YLim(1)) max(ax.XLim(2),ax.YLim(2))];
    plot(lims,lims,'k--','LineWidth',1);
    axis equal; xlim(lims); ylim(lims);
    
    xlabel('Actual'); ylabel('Predicted');
    title(['Output ',num2str(k),': ',outputNames{k}]);
    
    % Compute metrics for current output
    mseTrain_k = mean((YtrainPred(:,k) - YtrainAll(:,k)).^2);
    mseTest_k  = mean((YtestPred(:,k)  - Ytest(:,k)).^2);
    r2Train_k  = 1 - sum((YtrainPred(:,k)-YtrainAll(:,k)).^2) / ...
                       sum((YtrainAll(:,k)-mean(YtrainAll(:,k))).^2);
    r2Test_k   = 1 - sum((YtestPred(:,k)-Ytest(:,k)).^2) / ...
                       sum((Ytest(:,k)-mean(Ytest(:,k))).^2);
    
    % Annotation textbox
    text(0.05,0.95, ...
        {sprintf('Train  MSE = %.3g',mseTrain_k), ...
         sprintf('Train  R^2 = %.3f',r2Train_k), ...
         sprintf('Test   MSE = %.3g',mseTest_k), ...
         sprintf('Test   R^2 = %.3f',r2Test_k)}, ...
        'Units','normalized','FontSize',8,'VerticalAlignment','top', ...
        'BackgroundColor','w','EdgeColor','k','Margin',2);
    
    legend({'Train','Test','1:1'},'Location','southeast','FontSize',8);
end
sgtitle('Predicted vs. Actual Responses for All Outputs');


%% 7. Save trained model
save('BestANN.mat','bestNet','bestInfo','trainMSE','testMSE','trainR2','testR2');
disp('Best ANN model saved to BestANN.mat');
