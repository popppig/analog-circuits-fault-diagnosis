%pspice2data.m
function faultclass=pspice2data(path,row,column,order)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%功能:pspice数据导入matlab供classification
%输入:path-数据存放路径
%-----row要提取的数据行数，每个数据不同，因此需指定
%-----column数据列数，即样本量
%-----order样本分类标签，属第几类
%输出:第一行标签+后面数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pspicedata = importdata(path);
faultdata= pspicedata.data(1:row,2:end);
faultclass=[order*ones(1,column);faultdata];