%20170928_sallen_key.m
%统计量特征提取range,mean,std,Skewness,Kurtosis,Entropy  92%
% train/90:test/540  88%
%%
tic;
clc;
clear;
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
%%
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
%%
%合并,10train50test
data(:,1:10)=fault1(:,1:10);%181*10
data(:,11:20)=fault2(:,1:10);
data(:,21:30)=fault3(:,1:10);
data(:,31:40)=fault4(:,1:10);
data(:,41:50)=fault5(:,1:10);
data(:,51:60)=fault6(:,1:10);
data(:,61:70)=fault7(:,1:10);
data(:,71:80)=fault8(:,1:10);
data(:,81:90)=fault9(:,1:10);
%输入输出数据
test(:,1:50)=fault1(:,11:60);%181*50
test(:,51:100)=fault2(:,11:60);
test(:,101:150)=fault3(:,11:60);
test(:,151:200)=fault4(:,11:60);
test(:,201:250)=fault5(:,11:60);
test(:,251:300)=fault6(:,11:60);
test(:,301:350)=fault7(:,11:60);
test(:,351:400)=fault8(:,11:60);
test(:,401:450)=fault9(:,11:60);
%%
%输入输出数据
input_train=data(2:end,:);%input为136*90
output_train=data(1,:);%第一行

input_test=test(2:end,:);%input为136*450
output_test=test(1,:);%第一行

%%
%特征提取
for j=1:90
    s1=input_train(:,j);
    tezheng=feature2(s1);
    ss1(:,j)=tezheng';%5*450
end
eigenvalue_train=ss1;

for j=1:450
    s2=input_test(:,j);
    tezheng=feature2(s2);
    ss2(:,j)=tezheng';%5*450
end
eigenvalue_test=ss2;
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
%%
%libsvm
model=svmtrain(output_train',eigenvalue_train1','-t 2 -c 200 -g 0.1');
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
plot(output_train,Y1(:,4),'x');
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