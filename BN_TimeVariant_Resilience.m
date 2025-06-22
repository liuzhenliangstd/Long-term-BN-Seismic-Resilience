function BN_TimeVariant_Resilience()
% BN_TimeVariant_Resilience  基于ANN的区域桥梁地震韧性评估主程序
%   1) 多震级、多蒙特卡洛样本、多服役周期的地震场景生成
%   2) 利用GMPE计算各桥梁地震动强度
%   3) 通过训练好的ANN预测各桥梁关键构件地震响应
%   4) 根据脆弱性曲线判断损伤状态
%   5) 采用恢复函数模拟各桥梁功能恢复过程
%   6) 基于交通分配模型计算网络恢复功能演化
%   7) 计算5维韧性指标并绘制韧性分布

clear; clc;

% --- 0. 加载基础数据与模型 ---
load('BridgeInfo.mat');        % 桥梁基本信息结构体数组（含坐标、年龄、结构参数等）
load('LinkInfo.mat');          % 路段信息（含桥梁ID、自由流速度、容量等）
load('TrafficOD.mat');         % 交通OD矩阵
bestANN = load('BestANN.mat','bestNet');  
ANNnet = bestANN.bestNet;      % 训练好的ANN模型

% 极限状态阈值（示例从Excel导入）
AA1=importdata('Componentlimitstates.xlsx');
LS(:,:,1)=AA1.data(:,1:2); % pier column limit states
LS(:,:,2)=AA1.data(:,3:4); % longitudinal bearing deformation
LS(:,:,3)=AA1.data(:,5:6); % transverse bearing deformation
LS(:,:,4)=AA1.data(:,7:8); % passive abutment deformation
LS(:,:,5)=AA1.data(:,9:10); % active abutment deformation
LS(:,:,6)=AA1.data(:,11:12); % transverse abutment deformation

% --- 1. 参数设定 ---
magList = [6.0 6.5 7.0 7.5]; % 地震震级集合
epochs = 0:10:60;            % 服役期时间点(年)
Ngm = 103;                   % 每震级蒙特卡洛样本数量
t_grid = 0:1:30;             % 恢复时间窗口(小时)

Nbridge = length(BridgeInfo);
Nlink = length(LinkInfo);

% --- 2. 预分配存储 ---
R_all = zeros(length(epochs),length(magList),Ngm,5); % 韧性指标五维数组

% --- 3. 震害模拟主循环 ---
tic
for ie = 1:length(epochs)
    age = epochs(ie);
    % 根据服役期更新桥梁特征（示例函数，用户实现）
    XfeatBase = buildFeatureMatrix(BridgeInfo, age);
    
    for im = 1:length(magList)
        Mw = magList(im);
        
        % 计算各桥梁场地震动强度中值（log PGA）
        Rsite = hypot([BridgeInfo.X], [BridgeInfo.Y]); % 与震中距离
        Vs30_vec = [BridgeInfo.Vs30];
        logPGA_med = GMPE_BA08_vec(Mw,Rsite,Vs30_vec);
        
        % 蒙特卡洛地震动强度生成
        for g = 1:Ngm
            epsAleat = randn(1,Nbridge)*0.55; % Aleatory标准差假设0.55
            PGA = exp(logPGA_med).*exp(epsAleat);
            
            % 构造ANN输入特征矩阵：17桥梁 + 14地震动 = 31维
            Xnet = [XfeatBase, PGA'];
            
            % ANN预测地震响应（6个关键响应指标）
            Yhat = predict(ANNnet, Xnet);
            
            % --- 4. 损伤判定 ---
            Damage = zeros(Nbridge,6);
            for bri=1:Nbridge
                for comp=1:6
                    Damage(bri,comp) = FragilityBridge(Yhat(bri,comp), LS(:,:,comp));
                end
            end
            
            % 计算每桥梁综合损伤等级（示例简单取最大）
            DamageLevel = max(Damage,[],2); % 0-3等级
            
            % --- 5. 功能恢复模拟 ---
            % 功能恢复初始状态（残余功能率）根据损伤等级映射（示例）
            fac0 = damage2func(DamageLevel); 
            facT = ones(size(fac0)); % 完全恢复
            
            FuncBr = recoveryfunction(fac0, facT, 0.4, 3, t_grid);
            
            % --- 6. 交通网络恢复与功能计算 ---
            % 按时间点模拟交通流恢复
            Fnet = zeros(length(t_grid),5);
            for tt = 1:length(t_grid)
                % 路段功能因子映射（对应路段桥梁ID）
                facLink = FuncBr([LinkInfo.BridgeID],tt);
                % 增量交通分配计算路段流量及速度
                Flow = IncrementalUE(LinkInfo, LinkInfo.freeV'.*facLink, LinkInfo.capPcu'.*facLink, TrafficOD);
                
                % 五维韧性指标计算
                Fnet(tt,:) = calcResilienceMetrics(Flow, LinkInfo);
            end
            
            % --- 7. 积分计算韧性指标 ---
            R_all(ie,im,g,:) = trapz(t_grid,Fnet)./max(t_grid); % 时间积分归一化
        end
    end
end
toc

% --- 8. 结果示例绘图 ---
figure; 
R_em = squeeze(mean(R_all(:,:,:,1),3));
imagesc(magList, epochs, R_em);
colorbar; xlabel('Magnitude'); ylabel('Service Age (years)');
title('Mean Emergency-Response Resilience');

end

%% ====== 子函数区 ======

function Xfeat = buildFeatureMatrix(Bridge, age)
% 根据桥梁数据和服役时间构建ANN输入的17维特征矩阵 (Nbridge×17)
Nbridge = length(Bridge);
Xfeat = zeros(Nbridge,17);
for i = 1:Nbridge
    % 示例特征填充，可根据实际情况拓展
    Xfeat(i,1) = Bridge(i).Age + age;       % 实际服役年限
    Xfeat(i,2) = Bridge(i).Length;
    Xfeat(i,3) = Bridge(i).Width;
    Xfeat(i,4) = Bridge(i).Height;
    Xfeat(i,5) = Bridge(i).MaterialType;
    % ... (补充其他桥梁结构参数)
    % 假设剩余特征填零
end
end

function logPGA = GMPE_BA08_vec(Mw,R,Vs30)
% 基于Boore & Atkinson (2008) GMPE的简化向量化版本，计算各站点logPGA
% 公式来源: Boore & Atkinson (2008)
c1=-1.715; c2=0.5; c3=-0.1; c4=0.1; c5=1.3;
logPGA = c1 + c2*(Mw-6) + c3*(Mw-6).^2 - c5*log10(R+10^(c4*Mw));
logPGA = logPGA + 0.5*log10(760./Vs30);  % Vs30调整
end

function damageLevel = FragilityBridge(response, limitStates)
% 根据构件响应与极限状态判断损伤等级
% 输入：
%  response - 构件地震响应量
%  limitStates - 该构件对应的极限状态阈值矩阵(每行[min max])
% 输出：
%  damageLevel - 0(无损伤)到3(破坏)等级
damageLevel = 0;
for level = 1:size(limitStates,1)
    if response > limitStates(level,1) && response <= limitStates(level,2)
        damageLevel = level;
        break;
    elseif response > limitStates(end,2)
        damageLevel = size(limitStates,1);
    end
end
end

function fac = damage2func(damage)
% 损伤等级到功能率映射函数
% 0-无损伤→1，1-轻度→0.8，2-中度→0.5，3-重度破坏→0
lut = [1 0.8 0.5 0];
fac = lut(damage+1);
end

function TFt = recoveryfunction(delta0,deltar,omega,rou,t)
% 功能恢复函数，基于文献定义
% delta0: 初始残余功能率
% deltar: 最终恢复功能率
% omega, rou: 模型参数（0<=omega<=1，rou>1）
% t: 标准化时间点(0-1)
rt = zeros(size(t));
for i = 1:length(t)
    if t(i) <= omega
        rt(i) = omega^(1-rou)*t(i)^rou;
    else
        rt(i) = 1 - (1-omega)^(1-rou)*(1 - t(i))^rou;
    end
end
TFt = zeros(length(delta0), length(t));
for bri = 1:length(delta0)
    TFt(bri,:) = delta0(bri) + rt*(deltar(bri)-delta0(bri));
end
end

function Flow = IncrementalUE(LinkInfo, freeV_adj, cap_adj, TrafficOD)
% 增量用户均衡交通分配简易示例
% 输入路段自由流速度、容量调整后值，交通OD需求矩阵
% 输出路段流量估计
% 注意：此函数需根据实际交通流模型详细实现，以下为示例占位
Nlink = length(LinkInfo);
Flow = zeros(Nlink,5); % 五个指标占位
Flow(:,1) = cap_adj'; % 假设路段流量为容量调整值（示例）
% ... 实际交通分配算法需替换此处
end

function metrics = calcResilienceMetrics(Flow, LinkInfo)
% 计算五维韧性指标
% 输入路段流量及信息
% 输出五维韧性指标（示例）
metrics = zeros(1,5);
% 指标1：应急响应韧性 (travel time ratio)
metrics(1) = mean(Flow(:,1));  % 示例，真实按公式实现
% 指标2：社会稳定韧性
metrics(2) = mean(Flow(:,1))*0.9; % 示例
% 指标3：经济发展韧性
metrics(3) = mean(Flow(:,1))*0.8;
% 指标4：环境可持续韧性
metrics(4) = mean(Flow(:,1))*0.7;
% 指标5：交通便利韧性
metrics(5) = mean(Flow(:,1))*0.85;
% 请基于论文5节公式替换上述示例计算
end
