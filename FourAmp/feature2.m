%特征提取
function [shiyu]= feature2(x)

N = length(x);
%样本极差range
p1 = max(x)-min(x);
% 均值mean
p2 = mean(x);
% 标准差std
p3 = std(x);
% 偏斜度Skewness
p4 = skewness(x);
% 峭度Kurtosis
p5 = kurtosis(x);
%熵entropy
p6 = entropy(x);
shiyu=[p1 p2 p3 p4 p5 p6];
%shiyu=[p2 p3 p5 p7 p10 p11];