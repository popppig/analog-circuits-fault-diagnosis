%%
%信号特征提取16个时域、频域指标 均值、标准差、方差、 偏斜度指标、峭度指标、 峰-峰值、峰值、均方幅值、平均幅值、方根幅值、波形指标、峰值指标
%(Signal feature extraction 16 time domain, frequency domain mean, standard deviation, variance, skewness index, kurtosis index, peak- peak, mean square amplitude, average amplitude, square root amplitude, waveform indicators Peak Indicators)
%%
function u=tezhengtiqu16(r)
u=[];
for i=1:1:30
a=r(486*(i-1)+1:486*i,1);  %729*9
s=a;
u1=mean(s); % 均值
s=s-u1;
u2=std(s);% 标准差
u3=var(s); % 方差
u4=skewness(s); % 偏斜度指标
u5=kurtosis(s); % 峭度指标
u6=max(s)-min(s); % 峰-峰值
Xp=max(max(s), -min(s)); %峰值
Xrms=sqrt(mean(s.*s)); % 均方幅值
Xmean=mean(abs(s)); %平均幅值
Xr=mean(sqrt(abs(s)))*mean(sqrt(abs(s))); % 方根幅值
u7=Xrms/Xmean; %波形指标
u8=Xp/Xrms; % 峰值指标、脉冲指标、裕度指标、重心频率、频率方差、8个小波包能量巨
u9=Xp/Xmean;%脉冲指标、
u10=Xp/Xr;% 裕度指标
%均值、标准差、方差、 偏斜度指标、峭度指标、 峰-峰值、峰值、均方幅值、平均幅值、方根幅值、波形指标、峰值指标
uu=[u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,Xp,Xrms,Xmean,Xr];
u=cat(1,u,uu);
end
u;