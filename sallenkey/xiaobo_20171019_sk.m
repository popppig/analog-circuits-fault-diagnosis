%20170928_sallen_key.m
%5层小波包+PCA+svm
%随机选取样本,极小样本量82%
%%
tic;
clc;
clear;
disp('---start~');
%%
num_fault=9;%0-12
num_column=100;%样本量即列
num_all=num_fault*num_column;%所有样本总数
num_row=153;%数据行数即维数
num_train=num_all/2;
num_test=num_all-num_train;
%%
%数据导入并添加分类标签
fault0 = pspice2data('D:\matlab\sallenkey\data\f0.txt',num_row,num_column,0);
fault1 = pspice2data('D:\matlab\sallenkey\data\f1.txt',num_row,num_column,1);
fault2 = pspice2data('D:\matlab\sallenkey\data\f2.txt',num_row,num_column,2);
fault3 = pspice2data('D:\matlab\sallenkey\data\f3.txt',num_row,num_column,3);
fault4 = pspice2data('D:\matlab\sallenkey\data\f4.txt',num_row,num_column,4);
fault5 = pspice2data('D:\matlab\sallenkey\data\f5.txt',num_row,num_column,5);
fault6 = pspice2data('D:\matlab\sallenkey\data\f6.txt',num_row,num_column,6);
fault7 = pspice2data('D:\matlab\sallenkey\data\f7.txt',num_row,num_column,7);
fault8 = pspice2data('D:\matlab\sallenkey\data\f8.txt',num_row,num_column,8);
disp('---data import complete!');
%%
%数据合并
data=[fault0,fault1,fault2,fault3,fault4,fault5,fault6,fault7,fault8];
disp('---data combine complete!');
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
%5层小波包+方差
 for j=1:num_train
    s1=input_train(:,j);
    %用db1小波包对信号s1进行5层分解
    t=wpdec(s1,5,'db1','shannon');
    %下面对正常信号第三层各系数进行重构
    s1_0=wprcoef(t,[5,0]);                      %s1_0为210*1
    s1_1=wprcoef(t,[5,1]);                      %s1_1为210*1
    s1_2=wprcoef(t,[5,2]);                      %s1_2为210*1
    s1_3=wprcoef(t,[5,3]);                      %s1_3为210*1
    s1_4=wprcoef(t,[5,4]);                      %s1_4为210*1
    s1_5=wprcoef(t,[5,5]);                      %s1_5为210*1
    s1_6=wprcoef(t,[5,6]);                      %s1_6为210*1
    s1_7=wprcoef(t,[5,7]);                      %s1_7为210*1
    s1_8=wprcoef(t,[5,8]);                      %s1_8为210*1
    s1_9=wprcoef(t,[5,9]);                      %s1_9为210*1
    s1_10=wprcoef(t,[5,10]);                    %s1_10为210*1
    s1_11=wprcoef(t,[5,11]);                    %s1_11为210*1
    s1_12=wprcoef(t,[5,12]);                    %s1_12为210*1
    s1_13=wprcoef(t,[5,13]);                    %s1_13为210*1
    s1_14=wprcoef(t,[5,14]);                    %s1_14为210*1
    s1_15=wprcoef(t,[5,15]);                    %s1_15为210*1
    s1_16=wprcoef(t,[5,16]);                    %s1_16为210*1 
    s1_17=wprcoef(t,[5,17]);                    %s1_17为210*1
    s1_18=wprcoef(t,[5,18]);                    %s1_18为210*1
    s1_19=wprcoef(t,[5,19]);                    %s1_19为210*1
    s1_20=wprcoef(t,[5,20]);                    %s1_20为210*1
    s1_21=wprcoef(t,[5,21]);                    %s1_21为210*1
    s1_22=wprcoef(t,[5,22]);                    %s1_22为210*1
    s1_23=wprcoef(t,[5,23]);                    %s1_23为210*1
    s1_24=wprcoef(t,[5,24]);                    %s1_24为210*1
    s1_25=wprcoef(t,[5,25]);                    %s1_25为210*1
    s1_26=wprcoef(t,[5,26]);                    %s1_26为210*1
    s1_27=wprcoef(t,[5,27]);                    %s1_27为210*1
    s1_28=wprcoef(t,[5,28]);                    %s1_28为210*1
    s1_29=wprcoef(t,[5,29]);                    %s1_29为210*1
    s1_30=wprcoef(t,[5,30]);                    %s1_30为210*1
    s1_31=wprcoef(t,[5,31]);                    %s1_31为210*1
  
    %计算故障信号各重构系数的方差
    %计算正常信号各重构系数的方差
    s10=norm(s1_0);
    s11=norm(s1_1);
    s12=norm(s1_2);
    s13=norm(s1_3);
    s14=norm(s1_4);
    s15=norm(s1_5);
    s16=norm(s1_6);
    s17=norm(s1_7);
    s18=norm(s1_8);
    s19=norm(s1_9);
    s110=norm(s1_10);
    s111=norm(s1_11);
    s112=norm(s1_12);
    s113=norm(s1_13);
    s114=norm(s1_14);
    s115=norm(s1_15);
    s116=norm(s1_16);
    s117=norm(s1_17);
    s118=norm(s1_18);
    s119=norm(s1_19);
    s120=norm(s1_20);
    s121=norm(s1_21);
    s122=norm(s1_22);
    s123=norm(s1_23);
    s124=norm(s1_24);
    s125=norm(s1_25);
    s126=norm(s1_26);
    s127=norm(s1_27);
    s128=norm(s1_28);
    s129=norm(s1_29);
    s130=norm(s1_30);
    s131=norm(s1_31);             
    %向量ss1是针对信号s1构造的向量
    ss1(:,j)=[s10;s11;s12;s13;s14;s15;s16;s17;s18;s19;s110;s111;s112;s113;s114;s115;s116;s117;s118;s119;s120;s121;s122;s123;s124;s125;s126;s127;s128;s129;s130;s131];
end
eigenvalue_train=ss1 ;                                         %eigenvalue_train为32*500矩阵，训练集的特征值
% ss2=zeros(8,90);                              %ss2存储c测试的小波包分析特征向量8*190

for j=1:num_test
    s2=input_test(:,j);
    %用db1小波包对信号s1进行三层分解
    t=wpdec(s2,5,'db1','shannon');
    %下面对正常信号第三层各系数进行重构
    s2_0=wprcoef(t,[5,0]);                                     %s2_0为210*1 
    s2_1=wprcoef(t,[5,1]);                                     %s2_1为210*1
    s2_2=wprcoef(t,[5,2]);                                     %s2_2为210*1
    s2_3=wprcoef(t,[5,3]);                                     %s2_3为210*1
    s2_4=wprcoef(t,[5,4]);                                     %s2_4为210*1
    s2_5=wprcoef(t,[5,5]);                                     %s2_5为210*1
    s2_6=wprcoef(t,[5,6]);                                     %s2_6为210*1
    s2_7=wprcoef(t,[5,7]);                                     %s2_7为210*1
    s2_8=wprcoef(t,[5,8]);                                     %s2_8为210*1
    s2_9=wprcoef(t,[5,9]);                                     %s2_9为210*1  
    s2_10=wprcoef(t,[5,10]);                                   %s2_10为210*1
    s2_11=wprcoef(t,[5,11]);                                   %s2_11为210*1
    s2_12=wprcoef(t,[5,12]);                                   %s2_12为210*1
    s2_13=wprcoef(t,[5,13]);                                   %s2_13为210*1
    s2_14=wprcoef(t,[5,14]);                                   %s2_14为210*1
    s2_15=wprcoef(t,[5,15]);                                   %s2_15为210*1
    s2_16=wprcoef(t,[5,16]);                                   %s2_16为210*1
    s2_17=wprcoef(t,[5,17]);                                   %s2_17为210*1
    s2_18=wprcoef(t,[5,18]);                                   %s2_18为210*1
    s2_19=wprcoef(t,[5,19]);                                   %s2_19为210*1
    s2_20=wprcoef(t,[5,20]);                                   %s2_20为210*1
    s2_21=wprcoef(t,[5,21]);                                   %s2_21为210*1
    s2_22=wprcoef(t,[5,22]);                                   %s2_22为210*1
    s2_23=wprcoef(t,[5,23]);                                   %s2_23为210*1
    s2_24=wprcoef(t,[5,24]);                                   %s2_24为210*1
    s2_25=wprcoef(t,[5,25]);                                   %s2_25为210*1
    s2_26=wprcoef(t,[5,26]);                                   %s2_26为210*1
    s2_27=wprcoef(t,[5,27]);                                   %s2_27为210*1
    s2_28=wprcoef(t,[5,28]);                                   %s2_28为210*1
    s2_29=wprcoef(t,[5,29]);                                   %s2_29为210*1
    s2_30=wprcoef(t,[5,30]);                                   %s2_30为210*1
    s2_31=wprcoef(t,[5,31]);                                   %s2_31为210*1  
    
    %计算正常信号各重构系数的方差
    s20=norm(s2_0);
    s21=norm(s2_1);
    s22=norm(s2_2);
    s23=norm(s2_3);
    s24=norm(s2_4);
    s25=norm(s2_5);
    s26=norm(s2_6);
    s27=norm(s2_7);
    s28=norm(s2_8);
    s29=norm(s2_9);
    s210=norm(s2_10);
    s211=norm(s2_11);
    s212=norm(s2_12);
    s213=norm(s2_13);
    s214=norm(s2_14);
    s215=norm(s2_15);
    s216=norm(s2_16);
    s217=norm(s2_17);
    s218=norm(s2_18);
    s219=norm(s2_19);
    s220=norm(s2_20);
    s221=norm(s2_21);
    s222=norm(s2_22);
    s223=norm(s2_23);
    s224=norm(s2_24);
    s225=norm(s2_25);
    s226=norm(s2_26);
    s227=norm(s2_27);
    s228=norm(s2_28);
    s229=norm(s2_29);
    s230=norm(s2_30);
    s231=norm(s2_31);
    %向量ss2是针对信号s2构造的向量
    ss2(:,j)=[s20;s21;s22;s23;s24;s25;s26;s27;s28;s29;s210;s211;s212;s213;s214;s215;s216;s217;s218;s219;s220;s221;s222;s223;s224;s225;s226;s227;s228;s229;s230;s231];
end
eigenvalue_test=ss2 ;                            %eigenvalue_test为32*280矩阵，测试集特征值
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
%GridSearch寻优
[bestacc,bestc,bestg]=SVMcgForClass(output_train',eigenvalue_train1',-10,10,-10,10,5,0.5,0.5,4.5);
disp('Optimization complete!');
%%
%gaSVMcgForClass遗传参数优化
ga_option.maxgen = 100;
ga_option.sizepop = 20;
ga_option.cbound = [0,1000];
ga_option.gbound = [0,100];
ga_option.v = 5;
ga_option.ggap = 0.5;
[BestCVaccuracy,bestc,bestg,ga_option] = gaSVMcgForClass(output_train',eigenvalue_train1',ga_option);
disp('Optimization complete!');
%%
%psoSVMcgForClass参数寻优
[bestCVaccuracy,bestc,bestg]= psoSVMcgForClass(output_train',eigenvalue_train1')
%%
%libsvm
%model=svmtrain(output_train',eigenvalue_train1','-c 200,-g 1');%c增大到4的时候100%
cmd = ['-c ',num2str(bestc),'-g ',num2str(bestg)];
model=svmtrain(output_train',eigenvalue_train1',cmd);%c增大到4的时候100%
[predict_label, accuracy,prob_estimates] = svmpredict(output_test',eigenvalue_test1',model);

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
plot(output_train,Y1(:,4),'x');
%end;

%查看提取特征值的差异
figure(3);
hold on;
for j=1:32
    plot(output_train,ss1(j,:),'o');
end;