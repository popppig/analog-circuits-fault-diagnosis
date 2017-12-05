%20170928_sallen_key.m

tic;
clc;
clear;

filename1='D:\pspice\20170926-SallenKey-MC\excel数据\non-fault.xlsx';
filename2='D:\pspice\20170926-SallenKey-MC\excel数据\c1+.xlsx';
filename3='D:\pspice\20170926-SallenKey-MC\excel数据\c1-.xlsx';
filename4='D:\pspice\20170926-SallenKey-MC\excel数据\c2+.xlsx';
filename5='D:\pspice\20170926-SallenKey-MC\excel数据\c2-.xlsx';
filename6='D:\pspice\20170926-SallenKey-MC\excel数据\r2+.xlsx';
filename7='D:\pspice\20170926-SallenKey-MC\excel数据\r2-.xlsx';
filename8='D:\pspice\20170926-SallenKey-MC\excel数据\r3+.xlsx';
filename9='D:\pspice\20170926-SallenKey-MC\excel数据\r3+.xlsx';

sheet = 1;
xlRange = 'B2:CW137';
%读取excel数据
subset1 = xlsread(filename1,sheet,xlRange);
subset2 = xlsread(filename2,sheet,xlRange);
subset3 = xlsread(filename3,sheet,xlRange);
subset4 = xlsread(filename4,sheet,xlRange);
subset5 = xlsread(filename5,sheet,xlRange);
subset6 = xlsread(filename6,sheet,xlRange);
subset7 = xlsread(filename7,sheet,xlRange);
subset8 = xlsread(filename8,sheet,xlRange);
subset9 = xlsread(filename9,sheet,xlRange);
%第一行添加故障编码
fault1=[1*ones(1,100);subset1(1:136,1:100)];%137*100
fault2=[2*ones(1,100);subset1(1:136,1:100)];
fault3=[3*ones(1,100);subset1(1:136,1:100)];
fault4=[4*ones(1,100);subset1(1:136,1:100)];
fault5=[5*ones(1,100);subset1(1:136,1:100)];
fault6=[6*ones(1,100);subset1(1:136,1:100)];
fault7=[7*ones(1,100);subset1(1:136,1:100)];
fault8=[8*ones(1,100);subset1(1:136,1:100)];
fault9=[9*ones(1,100);subset1(1:136,1:100)];
%合并
data(:,1:100)=fault1;%137*900
data(:,101:200)=fault2;
data(:,201:300)=fault3;
data(:,301:400)=fault4;
data(:,401:500)=fault5;
data(:,501:600)=fault6;
data(:,601:700)=fault7;
data(:,701:800)=fault8;
data(:,801:900)=fault9;
%输入输出数据
input=data(2:137,:);%input为136*390
output=data(1,:);%第一行
%生成随机数
n=randperm(900); 
%500个数据为训练数据
input_train=input(:,n(1:500));                 %input_train为136*500矩阵;每一列代表一个样本
output_train=output(:,n(1:500));              %output_train为1*500矩阵。
%剩余400个数据为测试数据
input_test=input(:,n(501:end));                %input_test为136*400矩阵。
output_test=output(:,n(501:end));             %output_test为1*400矩阵。
%可修改为先小波再随机
for j=1:500
    s1=input_train(:,j);
    %用db1小波包对信号s1进行三层分解
    t=wpdec(s1,3,'db1','shannon');
    %下面对正常信号第三层各系数进行重构
    s130=wprcoef(t,[3,0]);
    s131=wprcoef(t,[3,1]);
    s132=wprcoef(t,[3,2]);
    s133=wprcoef(t,[3,3]);
    s134=wprcoef(t,[3,4]);
    s135=wprcoef(t,[3,5]);
    s136=wprcoef(t,[3,6]);
    s137=wprcoef(t,[3,7]);
  
    %计算故障信号各重构系数的方差
    s10=norm(s130);
    s11=norm(s131);
    s12=norm(s132);
    s13=norm(s133);
    s14=norm(s134);
    s15=norm(s135);
    s16=norm(s136);
    s17=norm(s137);
    %向量ss1是针对信号s1构造的向量
    ss1(:,j)=[s10;s11;s12;s13;s14;s15;s16;s17];
 end
eigenvalue_train=ss1 ;              %eigenvalue_train为8*200矩阵，训练特征值

% ss2=zeros(8,90);                              %ss2存储c测试的小波包分析特征向量8*190
for j=1:400
    s2=input_test(:,j);
    %用db1小波包对信号s1进行三层分解
    t=wpdec(s2,3,'db1','shannon');
    %下面对正常信号第三层各系数进行重构
    s230=wprcoef(t,[3,0]);
    s231=wprcoef(t,[3,1]);
    s232=wprcoef(t,[3,2]);
    s233=wprcoef(t,[3,3]);
    s234=wprcoef(t,[3,4]);
    s235=wprcoef(t,[3,5]);
    s236=wprcoef(t,[3,6]);
    s237=wprcoef(t,[3,7]);
    %计算正常信号各重构系数的方差,提取范数
    s20=norm(s130);
    s21=norm(s131);
    s22=norm(s132);
    s23=norm(s133);
    s24=norm(s134);
    s25=norm(s135);
    s26=norm(s136);
    s27=norm(s137);
    %向量ss2是针对信号s2构造的向量
    ss2(:,j)=[s20;s21;s22;s23;s24;s25;s26;s27];
end
eigenvalue_test=ss2 ;              %eigenvalue_test为8*190矩阵，测试特征值
%libsvm
model=svmtrain(output_train',eigenvalue_train');
[predict_label, accuracy, dec_values] = svmpredict(output_test',eigenvalue_test',model);

toc;