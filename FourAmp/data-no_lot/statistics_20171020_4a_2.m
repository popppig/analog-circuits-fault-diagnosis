%statistics_20171019.m
%统计量特征提取range,mean,std,Skewness,Kurtosis,Entropy
%随机选取样本,50%
%PCA仅提取了2个特征，诊断率64难以上去，可去掉
%需将PCA阈值提高至0.99
%%
clc;
clear;
disp('---start~');
%%
num_fault=13;%0-12
num_column=200;%样本量即列
num_all=num_fault*num_column;%所有样本总数
num_row=1000;%数据行数即维数
num_train=num_all/2;
num_test=num_all-num_train;
%%
%数据导入并添加分类标签
fault0 = pspice2data('D:\matlab\FourAmp\data-no_lot\f0.txt',num_row,num_column,0);
fault1 = pspice2data('D:\matlab\FourAmp\data-no_lot\f1.txt',num_row,num_column,1);
fault2 = pspice2data('D:\matlab\FourAmp\data-no_lot\f2.txt',num_row,num_column,2);
fault3 = pspice2data('D:\matlab\FourAmp\data-no_lot\f3.txt',num_row,num_column,3);
fault4 = pspice2data('D:\matlab\FourAmp\data-no_lot\f4.txt',num_row,num_column,4);
fault5 = pspice2data('D:\matlab\FourAmp\data-no_lot\f5.txt',num_row,num_column,5);
fault6 = pspice2data('D:\matlab\FourAmp\data-no_lot\f6.txt',num_row,num_column,6);
fault7 = pspice2data('D:\matlab\FourAmp\data-no_lot\f7.txt',num_row,num_column,7);
fault8 = pspice2data('D:\matlab\FourAmp\data-no_lot\f8.txt',num_row,num_column,8);
fault9 = pspice2data('D:\matlab\FourAmp\data-no_lot\f9.txt',num_row,num_column,9);
fault10 = pspice2data('D:\matlab\FourAmp\data-no_lot\f10.txt',num_row,num_column,10);
fault11 = pspice2data('D:\matlab\FourAmp\data-no_lot\f11.txt',num_row,num_column,11);
fault12 = pspice2data('D:\matlab\FourAmp\data-no_lot\f12.txt',num_row,num_column,12);
disp('---data import complete!');
%%
%数据合并
data=[fault0,fault1,fault2,fault3,fault4,fault5,fault6,fault7,fault8,fault9,fault10,fault11,fault12];
disp('---data combine complete!');
%%
%输入输出数据
input=data(2:end,:);
output=data(1,:);%第一行为标签
%生成随机数随机抽取
n=randperm(num_train+num_test); 
%num_train个数据为训练数据
input_train=input(:,n(1:num_train));                 
output_train=output(:,n(1:num_train));             
%剩余num_test个数据为测试数据
input_test=input(:,n((num_train+1):end));                
output_test=output(:,n((num_train+1):end));
disp('---data divide complete!');
%%
%特征提取-提取统计量
for j=1:num_train
    s1=input_train(:,j);
    tezheng=feature2(s1);
    ss1(:,j)=tezheng';
end
eigenvalue_train=ss1;

for j=1:num_test
    s2=input_test(:,j);
    tezheng=feature2(s2);
    ss2(:,j)=tezheng';
end
eigenvalue_test=ss2;
disp('---feature extraction complete!');
%%
%主元分析
%仅提取了2个参数，暂时不用PCA
[nx1,ny1]=size(eigenvalue_train');                         %eigenvalue_train'为500*32
[nx2,ny2]=size(eigenvalue_test');                          %eigenvalue_test'为280*32
eigenvalue=[eigenvalue_train';eigenvalue_test'];           %eigenvalue为280*32
Y=myPCA(eigenvalue);                                       %Y为780*2矩阵
Y1=Y(1:nx1,:);                                             %Y1为500*2矩阵
Y2=Y(nx1+1:end,:);                                         %Y2为280*2矩阵

eigenvalue_train1=Y1';                                     %eigenvalue_train1为2*500
eigenvalue_test1=Y2';                                      %eigenvalue_test1为2*280
disp('---feature selection complete!');

%coeff_train = pca(eigenvalue_train); %matlab自带pca实现
%coeff_test = pca(eigenvalue_test);
%%
%GridSearch寻优
tic;
[bestacc,bestc,bestg]=SVMcgForClass(output_train',eigenvalue_train',-8,10,-10,8,5,1,1,4.5);
toc;
disp('Optimization complete!');
%%
%gaSVMcgForClass遗传算法参数优化
tic;
ga_option.maxgen = 100;
ga_option.sizepop = 20;
ga_option.cbound = [0,1000];
ga_option.gbound = [0,100];
ga_option.v = 3;
ga_option.ggap = 0.5;
[BestCVaccuracy,bestc,bestg,ga_option] = gaSVMcgForClass(output_train',eigenvalue_train1',ga_option);
toc;
disp('Optimization complete!');
%%
%psoSVMcgForClass粒子算法参数寻优
tic;
[bestCVaccuracy,bestc,bestg]= psoSVMcgForClass(output_train',eigenvalue_train1')
toc;
disp('Optimization complete!');
%%
%libsvm
%model=svmtrain(output_train',coeff1,'-c 200,-g 1');[predict_label, accuracy,prob_estimates] = svmpredict(output_test',coeff2,model);

cmd = ['-c ',num2str(bestc),'-g ',num2str(bestg)];
model=svmtrain(output_train',eigenvalue_train1',cmd);%c增大到4的时候100%
[predict_label, accuracy,prob_estimates] = svmpredict(output_test',eigenvalue_test1',model);


%%
%%结果分析
%测试集的实际分类和预测分类图
%通过图可以看出
figure(1);
hold on;
plot(output_test,'o');
plot(predict_label,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on;


%%
%查看提取特征值的差异
figure(2);
hold on;
%for j=1:3
plot(output_train,Y1(:,1),'o');
plot(output_train,Y1(:,2),'+');
plot(output_train,Y1(:,3),'x');
%end;

%%
%查看提取特征值的差异
figure(3);
hold on;
%for j=1:3
plot(output_train,ss1(1,:),'o');
plot(output_train,ss1(2,:),'+');
plot(output_train,ss1(3,:),'x');
plot(output_train,ss1(4,:),'*');
plot(output_train,ss1(5,:),'s');
plot(output_train,ss1(6,:),'d');
%end;