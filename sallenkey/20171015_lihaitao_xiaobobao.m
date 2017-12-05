%20170928_sallen_key.m
%lihaitao数据
%5层小波包+PCA+svm
%去掉归一化不影响数据准确度
%%
tic;
clc;
clear;
format long

% filename1='D:\pspice\20170926-SallenKey-MC\excel数据\non-fault.xlsx';
% filename2='D:\pspice\20170926-SallenKey-MC\excel数据\c1+.xlsx';
% filename3='D:\pspice\20170926-SallenKey-MC\excel数据\c1-.xlsx';
% filename4='D:\pspice\20170926-SallenKey-MC\excel数据\c2+.xlsx';
% filename5='D:\pspice\20170926-SallenKey-MC\excel数据\c2-.xlsx';
% filename6='D:\pspice\20170926-SallenKey-MC\excel数据\r2+.xlsx';
% filename7='D:\pspice\20170926-SallenKey-MC\excel数据\r2-.xlsx';
% filename8='D:\pspice\20170926-SallenKey-MC\excel数据\r3+.xlsx';
% filename9='D:\pspice\20170926-SallenKey-MC\excel数据\r3-.xlsx';
% 
% sheet = 1;
% xlRange = 'B2:CW137';
% %读取excel数据
% subset1 = xlsread(filename1,sheet,xlRange);
% subset2 = xlsread(filename2,sheet,xlRange);
% subset3 = xlsread(filename3,sheet,xlRange);
% subset4 = xlsread(filename4,sheet,xlRange);
% subset5 = xlsread(filename5,sheet,xlRange);
% subset6 = xlsread(filename6,sheet,xlRange);
% subset7 = xlsread(filename7,sheet,xlRange);
% subset8 = xlsread(filename8,sheet,xlRange);
% subset9 = xlsread(filename9,sheet,xlRange);
%%
c1=load('d:\matlab\sallenkey\李海涛数据\MC60\f0_normal.txt','rt');   %c1为250*61
c2=load('d:\matlab\sallenkey\李海涛数据\MC60\f1_r3_rise.txt','rt');  %c2为245*61
c3=load('d:\matlab\sallenkey\李海涛数据\MC60\f2_r3_fall.txt','rt');  %c3为237*61
c4=load('d:\matlab\sallenkey\李海涛数据\MC60\f3_r8_rise.txt','rt');  %c4为257*61
c5=load('d:\matlab\sallenkey\李海涛数据\MC60\f4_r8_fall.txt','rt');  %c5为240*61
c6=load('d:\matlab\sallenkey\李海涛数据\MC60\f5_r16_rise.txt','rt'); %c6为303*61
c7=load('d:\matlab\sallenkey\李海涛数据\MC60\f6_r16_fall.txt','rt'); %c7为185*61
c8=load('d:\matlab\sallenkey\李海涛数据\MC60\f7_c1_rise.txt','rt');  %c8为249*61
c9=load('d:\matlab\sallenkey\李海涛数据\MC60\f8_c1_fall.txt','rt');  %c9为251*61

%第一行添加故障编码
fault1=[1*ones(1,60);c1(1:180,2:61)];%181*60
fault2=[2*ones(1,60);c2(1:180,2:61)];
fault3=[3*ones(1,60);c3(1:180,2:61)];
fault4=[4*ones(1,60);c4(1:180,2:61)];
fault5=[5*ones(1,60);c5(1:180,2:61)];
fault6=[6*ones(1,60);c6(1:180,2:61)];
fault7=[7*ones(1,60);c7(1:180,2:61)];
fault8=[8*ones(1,60);c8(1:180,2:61)];
fault9=[9*ones(1,60);c9(1:180,2:61)];
%合并,30train30test
data(:,1:30)=fault1(:,1:30);%181*30
data(:,31:60)=fault2(:,1:30);
data(:,61:90)=fault3(:,1:30);
data(:,91:120)=fault4(:,1:30);
data(:,121:150)=fault5(:,1:30);
data(:,151:180)=fault6(:,1:30);
data(:,181:210)=fault7(:,1:30);
data(:,211:240)=fault8(:,1:30);
data(:,241:270)=fault9(:,1:30);
%输入输出数据
test(:,1:30)=fault1(:,31:60);%181*30
test(:,31:60)=fault2(:,31:60);
test(:,61:90)=fault3(:,31:60);
test(:,91:120)=fault4(:,31:60);
test(:,121:150)=fault5(:,31:60);
test(:,151:180)=fault6(:,31:60);
test(:,181:210)=fault7(:,31:60);
test(:,211:240)=fault8(:,31:60);
test(:,241:270)=fault9(:,31:60);
%%
%输入输出数据
input_train=data(2:end,:);%input为180*270
output_train=data(1,:);%第一行

input_test=test(2:end,:);%input为180*270
output_test=test(1,:);%第一行
%%
 for j=1:270
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
for j=1:270
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
%%
%主元分析

[nx1,ny1]=size(eigenvalue_train');                         %eigenvalue_train'为500*32
[nx2,ny2]=size(eigenvalue_test');                          %eigenvalue_test'为280*32
eigenvalue=[eigenvalue_train';eigenvalue_test'];           %eigenvalue为280*32
Y=myPCA(eigenvalue);                                       %Y为780*2矩阵
Y1=Y(1:nx1,:);                                             %Y1为500*2矩阵
Y2=Y(nx1+1:end,:);                                         %Y2为280*2矩阵

eigenvalue_train=Y1';                                     %eigenvalue_train1为2*500
eigenvalue_test=Y2'; %eigenvalue_test1为2*280

%%
%libsvm
model=svmtrain(output_train',eigenvalue_train','-c 2,-g 1');%c增大到4的时候100%
[predict_label, accuracy, prob_estimates] = svmpredict(output_test',eigenvalue_test',model);

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
plot(output_train,Y1(:,5),'s');
%end;

%查看提取特征值的差异
figure(3);
hold on;
for j=1:32
    plot(output_train,ss1(j,:),'o');
end;