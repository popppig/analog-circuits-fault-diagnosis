function [bestCVaccuarcy,bestc,bestg,sca_option] = scaSVMcgForClass(train_label,train,sca_option)
% 基于FarutoUltimate3.1中psoSVMcgForClass修改
% psoSVMcgForClass
%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto
%last modified 2011.06.08
% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2011. 
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm
%% 参数初始化
if nargin == 2
    sca_option = struct('maxgen',200,'sizepop',20,'v',5, ...
        'popcmax',10^2,'popcmin',10^(-1),'popgmax',10^3,'popgmin',10^(-2));
end
% c1:初始为1.5,pso参数局部搜索能力
% c2:初始为1.7,pso参数全局搜索能力
% maxgen:初始为200,最大进化数量
% sizepop:初始为20,种群最大数量
% k:初始为0.6(k belongs to [0.1,1.0]),速率和x的关系(V = kX)
% wV:初始为1(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
% wP:初始为1,种群更新公式中速度前面的弹性系数
% v:初始为3,SVM Cross Validation参数
% popcmax:初始为100,SVM 参数c的变化的最大值.
% popcmin:初始为0.1,SVM 参数c的变化的最小值.
% popgmax:初始为1000,SVM 参数g的变化的最大值.
% popgmin:初始为0.01,SVM 参数c的变化的最小值.

eps = 1e-1;

%% 产生初始粒子和速度
for i=1:sca_option.sizepop%对每个粒子来说
    
    % 随机产生种群和速度
    pop(i,1) = (sca_option.popcmax-sca_option.popcmin)*rand+sca_option.popcmin;
    pop(i,2) = (sca_option.popgmax-sca_option.popgmin)*rand+sca_option.popgmin;

    % 计算初始适应度
    cmd = ['-v ',num2str(sca_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
    fitness(i) = svmtrain(train_label, train, cmd);
    fitness(i) = -fitness(i);
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值及其在局部极值中的位置
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点坐标
local_x=pop;    % 个体极值点坐标初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,sca_option.maxgen);

%% 迭代寻优
for i=1:sca_option.maxgen
    
    a = 2;%单调递减至0
    r1=a-i*((a)/sca_option.maxgen); % r1 decreases linearly from a to 0
    
    for j=1:sca_option.sizepop
        
        %种群更新
        % Update r2, r3, and r4 for Eq. (3.3)
        r2=(2*pi)*rand();
        r3=2*rand;
        r4=rand();

        % Eq. (3.3)
        if r4<0.5
            % Eq. (3.1)
            pop(j,:)= pop(j,:)+(r1*sin(r2)*abs(r3*global_x-pop(j,:)));
        else
            % Eq. (3.2)
            pop(j,:)= pop(j,:)+(r1*cos(r2)*abs(r3*global_x-pop(j,:)));
        end
        
        if pop(j,1) > sca_option.popcmax
            pop(j,1) = sca_option.popcmax;
        end
        if pop(j,1) < sca_option.popcmin
            pop(j,1) = sca_option.popcmin;
        end
        if pop(j,2) > sca_option.popgmax
            pop(j,2) = sca_option.popgmax;
        end
        if pop(j,2) < sca_option.popgmin
            pop(j,2) = sca_option.popgmin;
        end
        
%         % 自适应粒子变异
%         if rand>0.5
%             k=ceil(2*rand);
%             if k == 1
%                 pop(j,k) = (20-1)*rand+1;
%             end
%             if k == 2
%                 pop(j,k) = (sca_option.popgmax-sca_option.popgmin)*rand + sca_option.popgmin;
%             end
%         end
        
        %适应度值
        cmd = ['-v ',num2str(sca_option.v),' -c ',num2str( pop(j,1) ),' -g ',num2str( pop(j,2) )];
        fitness(j) = svmtrain(train_label, train, cmd);
        fitness(j) = -fitness(j);
        
%         cmd_temp = ['-c ',num2str( pop(j,1) ),' -g ',num2str( pop(j,2) )];
%         model = svmtrain(train_label, train, cmd_temp);
        
        if fitness(j) >= -65
            continue;
        end
        
        
        %群体最优更新
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
        
        if abs( fitness(j)-global_fitness )<=eps && pop(j,1) < global_x(1)
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
        
    end
    
    fit_gen(i) = global_fitness;
    avgfitness_gen(i) = sum(fitness)/sca_option.sizepop;
end

%% 结果分析
figure;
hold on;
plot(-fit_gen,'r*-');
plot(-avgfitness_gen,'o-');
legend('最佳适应度','平均适应度',3);
xlabel('进化代数','FontSize',12);
ylabel('适应度','FontSize',12);
grid on;

bestc = global_x(1);
bestg = global_x(2);
bestCVaccuarcy = -fit_gen(sca_option.maxgen);

line1 = '适应度曲线Accuracy[PSOmethod]';
line2 = ['(终止代数=', ...
    num2str(sca_option.maxgen),',种群数量pop=', ...
    num2str(sca_option.sizepop),')'];
% line3 = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVAccuracy=',num2str(bestCVaccuarcy),'%'];
% title({line1;line2;line3},'FontSize',12);
title({line1;line2},'FontSize',12);


