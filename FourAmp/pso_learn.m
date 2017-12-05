c1=1.5;
c2=1.7;
maxgen=200;
sizepop=20;
k=0.6;
wV=1;
wP=1;
v=3;
popcmax=100;
popcmin=0.1;
popgmax=1000;
popgmin=0.01;

Vcmax = pso_option.k*pso_option.popcmax;%搜索速度
Vcmin = -Vcmax ;
Vgmax = pso_option.k*pso_option.popgmax;
Vgmin = -Vgmax ;

eps = 1e-1;

%% 产生初始粒子和速度
for i=1:pso_option.sizepop
    
    % 随机产生种群和速度
    pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;%rand  0-1正态分布
    pop(i,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
    V(i,1)=Vcmax*rands(1,1);
    V(i,2)=Vgmax*rands(1,1);
    
    % 计算初始适应度
    cmd = ['-v ',num2str(pso_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) )];
    fitness(i) = svmtrain(train_label, train, cmd);
    fitness(i) = -fitness(i);%求最小值
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值和序号
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点pop(c,g)
local_x=pop;    % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,pso_option.maxgen);