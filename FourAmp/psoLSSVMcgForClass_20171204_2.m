function [bestCVaccuarcy,bestgam,bestsig2,pso_option] = psoLSSVMcgForClass(trainset,trainset_label,pso_option)
% psoLSSVMcgForClass 
%by Tangxiaobiao  QQ 444646122 blog.sina.com.cn/lssvm
% 2010.05.31
% 更改部分冗余表示by KingsleyChu
% 2017.12.04
%% 参数初始化
if nargin == 2
    pso_option = struct('c1',1.5,'c2',1.7,'maxgen',20,'sizepop',20, ...
        'k',0.6,'wV',1,'wP',1,'v',5, ...
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
% Yc=Yc;
Vcmax = pso_option.k*pso_option.popcmax;
Vcmin = -Vcmax ;
Vgmax = pso_option.k*pso_option.popgmax;
Vgmin = -Vgmax ;

eps = 10^(-3);
[Yc,codebook,old_codebook] = code(trainset_label,'code_MOC');
%% 产生初始粒子和速度
for i=1:pso_option.sizepop
    
    % 随机产生种群和速度
    pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
    pop(i,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
    V(i,1)=Vcmax*rands(1);
    V(i,2)=Vgmax*rands(1);
    
    % 计算初始适应度
    gam=pop(i,1);sig2=pop(i,2);
    model=initlssvm(trainset,Yc,'c',gam,sig2,'RBF_kernel');
    model.kernel_pars=sig2;
    model=trainlssvm(model);
    Yd0=simlssvm(model,trainset);
    predict_label = code(Yd0,old_codebook,[],codebook);%解码分类结果
    testnum=size(trainset_label,1);
    right=sum(trainset_label==predict_label);
%     n = sum(trainset_label~=predict_label);
%     fitness(i) = (1-n/prod(size(trainset_label)))*100;%prod计算数组元素的连乘积,因label只有1列，即为元素个数
    fitness(i) = (right/testnum)*100;
	fitness(i) = -fitness(i);
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点
local_x=pop;    % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,pso_option.maxgen);

%% 迭代寻优
for i=1:pso_option.maxgen
    
    for j=1:pso_option.sizepop
        
        %速度更新
        V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        if V(j,1) > Vcmax
            V(j,1) = Vcmax;
        end
        if V(j,1) < Vcmin
            V(j,1) = Vcmin;
        end
        if V(j,2) > Vgmax
            V(j,2) = Vgmax;
        end
        if V(j,2) < Vgmin
            V(j,2) = Vgmin;
        end
        
        %种群更新
        pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
        if pop(j,1) > pso_option.popcmax
            pop(j,1) = pso_option.popcmax;
        end
        if pop(j,1) < pso_option.popcmin
            pop(j,1) = pso_option.popcmin;
        end
        if pop(j,2) > pso_option.popgmax
            pop(j,2) = pso_option.popgmax;
        end
        if pop(j,2) < pso_option.popgmin
            pop(j,2) = pso_option.popgmin;
        end
        
        % 自适应粒子变异
        if rand>0.5
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (20-1)*rand+1;
            end
            if k == 2
                pop(j,k) = (pso_option.popgmax-pso_option.popgmin)*rand + pso_option.popgmin;
            end
        end
        
        %适应度值
     gam=pop(j,1);sig2=pop(j,2);
     model=initlssvm(trainset,Yc,'c',gam,sig2,'RBF_kernel');
     model=trainlssvm(model);
     Yd0=simlssvm(model,trainset);
     predict_label = code(Yd0,old_codebook,[],codebook);%解码分类结果
     testnum=size(trainset_label,1);
     right=sum(trainset_label==predict_label);
     n = sum(sum(trainset_label~=predict_label));
     fitness(j) = (1-n/prod(size(trainset_label)))*100;
     fitness(j) = -fitness(j);

     gam=pop(j,1);sig2=pop(j,2);
     model=initlssvm(trainset,Yc,'c',gam,sig2,'RBF_kernel');
     model=trainlssvm(model); 
     
        if fitness(j) >= -65
            continue;
        end
        
        %个体最优更新
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if abs( fitness(j)-local_fitness(j) )<=eps && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
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
    avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
end

%% 结果分析
figure;
hold on;
plot(-fit_gen,'r*-','LineWidth',1.5);
plot(-avgfitness_gen,'o-','LineWidth',1.5);
legend('最佳适应度','平均适应度',3);
xlabel('进化代数','FontSize',10);
ylabel('适应度','FontSize',10);
grid on;

% print -dtiff -r600 pso

bestgam = global_x(1);
bestsig2 = global_x(2);
bestCVaccuarcy = -fit_gen(pso_option.maxgen);

line1 = 'PSO optimize LSSVM-Classification model';
line2 = ['(参数c1=',num2str(pso_option.c1), ...
    ',c2=',num2str(pso_option.c2),',终止代数=', ...
    num2str(pso_option.maxgen),',种群数量pop=', ...
    num2str(pso_option.sizepop),')'];
line3 = ['Best gam=',num2str(bestgam),' sig2=',num2str(bestsig2), ...
    ' PSO-cvaccuracy=',num2str(bestCVaccuarcy),'%'];
title({line1;line2;line3},'FontSize',10);


