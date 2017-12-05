%20170928_sallen_key.m
%统计量特征提取range,mean,std,Skewness,Kurtosis,Entropy
%减少训练集数目train1:9test
%随机选取样本,极小样本量94%
%%
tic;
clc;
clear;
disp('start~');
%%
num_train=20;
num_test=880;
%%
filename1='D:\pspice\20170926-SallenKey-MC\excel数据\non-fault.xlsx';
filename2='D:\pspice\20170926-SallenKey-MC\excel数据\c1+.xlsx';
filename3='D:\pspice\20170926-SallenKey-MC\excel数据\c1-.xlsx';
filename4='D:\pspice\20170926-SallenKey-MC\excel数据\c2+.xlsx';
filename5='D:\pspice\20170926-SallenKey-MC\excel数据\c2-.xlsx';
filename6='D:\pspice\20170926-SallenKey-MC\excel数据\r2+.xlsx';
filename7='D:\pspice\20170926-SallenKey-MC\excel数据\r2-.xlsx';
filename8='D:\pspice\20170926-SallenKey-MC\excel数据\r3+.xlsx';
filename9='D:\pspice\20170926-SallenKey-MC\excel数据\r3-.xlsx';

sheet = 1;
xlRange = 'B2:CW137';
%读取excel数据
subset1 = xlsread(filename1,sheet,xlRange);%136*100
subset2 = xlsread(filename2,sheet,xlRange);
subset3 = xlsread(filename3,sheet,xlRange);
subset4 = xlsread(filename4,sheet,xlRange);
subset5 = xlsread(filename5,sheet,xlRange);
subset6 = xlsread(filename6,sheet,xlRange);
subset7 = xlsread(filename7,sheet,xlRange);
subset8 = xlsread(filename8,sheet,xlRange);
subset9 = xlsread(filename9,sheet,xlRange);
disp('data import end')
%%
%第一行添加故障编码
fault1=[1*ones(1,100);subset1(1:136,1:100)];%137*100
fault2=[2*ones(1,100);subset2(1:136,1:100)];
fault3=[3*ones(1,100);subset3(1:136,1:100)];
fault4=[4*ones(1,100);subset4(1:136,1:100)];
fault5=[5*ones(1,100);subset5(1:136,1:100)];
fault6=[6*ones(1,100);subset6(1:136,1:100)];
fault7=[7*ones(1,100);subset7(1:136,1:100)];
fault8=[8*ones(1,100);subset8(1:136,1:100)];
fault9=[9*ones(1,100);subset9(1:136,1:100)];
%合并,50train50test
data(:,1:100)=fault1;%137*900
data(:,101:200)=fault2;
data(:,201:300)=fault3;
data(:,301:400)=fault4;
data(:,401:500)=fault5;
data(:,501:600)=fault6;
data(:,601:700)=fault7;
data(:,701:800)=fault8;
data(:,801:900)=fault9;

%%
%输入输出数据
input=data(2:end,:);%input为136*450
output=data(1,:);%第一行
%生成随机数
n=randperm(num_train+num_test); 
%500个数据为训练数据
input_train=input(:,n(1:num_train));                 %input_train为136*500矩阵;每一列代表一个样本
output_train=output(:,n(1:num_train));              %output_train为1*500矩阵。
%剩余400个数据为测试数据
input_test=input(:,n((num_train+1):end));                %input_test为136*400矩阵。
output_test=output(:,n((num_train+1):end));             %output_test为1*400矩阵。

%%
%特征提取
for j=1:num_train
    s1=input_train(:,j);
    tezheng=feature2(s1);
    ss1(:,j)=tezheng';%5*450
end
eigenvalue_train=ss1;

for j=1:num_test
    s2=input_test(:,j);
    tezheng=feature2(s2);
    ss2(:,j)=tezheng';%5*450
end
eigenvalue_test=ss2;
disp('feature extraction complete!');
%%
%主元分析

[nx1,ny1]=size(eigenvalue_train');                         %eigenvalue_train'为500*32
[nx2,ny2]=size(eigenvalue_test');                          %eigenvalue_test'为280*32
eigenvalue=[eigenvalue_train';eigenvalue_test'];           %eigenvalue为280*32
Y=myPCA(eigenvalue);                                       %Y为780*2矩阵
Y1=Y(1:nx1,:);                                             %Y1为500*2矩阵
Y2=Y(nx1+1:end,:);                                         %Y2为280*2矩阵

eigenvalue_train1=Y1';                                     %eigenvalue_train1为2*500
eigenvalue_test1=Y2';                                      %eigenvalue_test1为2*280
disp('feature selection complete!');
%%
%libsvm
model=svmtrain(output_train',eigenvalue_train1','-t 2 -c 100 -g 1');
[predict_label, accuracy, dec_values] = svmpredict(output_test',eigenvalue_test1',model);

toc;

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
plot(output_train,Y1(:,3),'*');
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