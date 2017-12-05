%特征提取
%20171020增加p7-p10
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
%最大值max
p7 = max(x);
%最小值min
p8 = max(x);
%样本中位数
p9 = median(x);
%样本3阶中心距,E{[X-E(X)]^k}
p10=moment(x,3);
shiyu=[p1 p2 p3 p4 p5 p6 p7 p8 p9 p10];
%shiyu=[p2 p3 p5 p7 p10 p11];