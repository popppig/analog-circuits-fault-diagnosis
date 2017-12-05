function [eigvector, eigvalue,Y] = myKPCA(X,r,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kernel Principal Component Analysis
% [eigvector, eigvalue,Y] = KPCA(X,r,opts)
% Input:
% X: d*N data matrix;Each column vector of X is a sample vector.
% r: Dimensionality of reduced space (default: d)
% opts:   Struct value in Matlab. The fields in options that can be set:           
%         KernelType  -  Choices are:
%                  'Gaussian'      - exp{-gamma(|x-y|^2)}
%                  'Polynomial'    - (x'*y)^d
%                  'PolyPlus'      - (x'*y+1)^d
%         gamma       -  parameter for Gaussian kernel
%         d           -  parameter for polynomial kernel
%
% Output:
% eigvector: N*r matrix;Each column is an embedding function, for a new
%            data point (column vector) x,  y = eigvector'*K(x,:)'
%            will be the embedding result of x.
%            K(x,:) = [K(x1,x),K(x2,x),...K(xN,x)]
% eigvalue: The sorted eigvalue of KPCA eigen-problem.
% Y       : Data matrix after the nonlinear transform

if nargin<1
  error('Not enough input arguments.')
end
[d,N]=size(X);
if nargin<2
  r=d;
end
%% Ensure r is not bigger than d
if r>d
    r=d;
end;
% Construct the Kernel matrix K
K =ConstructKernelMatrix(X,[],opts);
% Centering kernel matrix
One_N=ones(N)./N;
Kc = K - One_N*K - K*One_N + One_N*K*One_N;         %Kc替代K，满足去均值的要求
clear One_N;
% Solve the eigenvalue problem N*lamda*alpha = K*alpha
if N>1000 && r
    % using eigs to speed up!
    opts.disp=0;
    [eigvector, eigvalue] = eigs(Kc,r,'la',opts);   %求Kc的特征值和特征向量，其中'la'表示最大特征值，r表示特征值的个数
    eigvalue = diag(eigvalue);                      %得到特征值的行向量
else
    [eigvector, eigvalue] = eig(Kc);                %求Kc的特征值和特征向量                               
    eigvalue = diag(eigvalue);                      %得到特征值的行向量
    [junk, index] = sort(-eigvalue);                %排序sort一般是从小到大排序，取负的目的是得到从大到小的排序             
    eigvalue = eigvalue(index);                     %按从大到小排序特征值
    eigvector = eigvector(:,index);                 %与特征值对应排序特征向量
end
if r < length(eigvalue)
    eigvalue = eigvalue(1:r);
    eigvector = eigvector(:,1:r);
end
% Only reserve the eigenvector with nonzero eigenvalues保留非零特征值对应的特征向量
maxEigValue = max(abs(eigvalue));                   %取绝对值最大的特征值
eigIdx = find(abs(eigvalue)/maxEigValue < 1e-6);    %找到绝对值特征值与最大特征值之比小于1e-6的特征值的索引
eigvalue (eigIdx) = [];                             %把找到的特征值赋空值
eigvector (:,eigIdx) = [];                          %把找到的特征值对应的特征向量赋空值

% Normalizing eigenvector归一化特征向量
% for i=1:m
for i=1:length(eigvalue)
    eigvector(:,i)=eigvector(:,i)/sqrt(eigvalue(i));
end;
%%%%%%%%%%%
%我加的程序
% sum_latent=sum(eigvalue);
% temp=0;
% con=0;
% m=0;
% for i=1:d
%     if con<0.8
%         temp=temp+eigvalue(i);
%         con=temp/sum_latent;
%         m=m+1;
%     else
%         break;
%     end
% end
% eigvector(:,m+1:d)=[];
%%%%%%%%%%%%


if nargout >= 3
    % Projecting the data in lower dimensions
    Y = eigvector'*K;
end
 
function K=ConstructKernelMatrix(X,Y,opts)
%
% function K=ConstructKernelMatrix(X,Y,opts)
% Usage:
%   opts.KernelType='Gaussian';
% K = ConstructKernelMatrix(X,[],opts)
%   K = ConstructKernelMatrix(X,Y,opts)
%
% Input:
% X: d*N data matrix;Each column vector of X is a sample vector.
% Y: d*M data matrix;Each column vector of Y is a sample vector.
% opts:   Struct value in Matlab. The fields in options that can be set:                
%         KernelType  -  Choices are:
%                  'Gaussian'      - exp{-gamma(|x-y|^2)}
%                  'Polynomial'    - (x'*y)^d
%                  'PolyPlus'      - (x'*y+1)^d
%         gamma       -  parameter for Gaussian kernel
%         d           -  parameter for polynomial kernel
% Output:
% K N*N or N*M matrix
if nargin<1
  error('Not enough input arguments.')
end
if (~exist('opts','var'))        %看opts变量是否存在，如果不存在则重新赋值
   opts = [];
else
   if ~isstruct(opts)            %如果opts不是结构体，则提示错误
       error('parameter error!');
   end
end
N=size(X,2);
if isempty(Y)     %如果Y是空，则执行下面
    K=zeros(N,N);
else
    M=size(Y,2);                 %返回Y的列数
    if size(X,1)~=size(Y,1)      %返回行数    
        error('Matrixes X and Y should have the same row dimensionality!');
    end;
    K=zeros(N,M);
end;
%=================================================
if ~isfield(opts,'KernelType')                 
    opts.KernelType = 'Gaussian';   
end
switch lower(opts.KernelType)
    case {lower('Gaussian')}        %  exp{-gamma(|x-y|^2)}
        if ~isfield(opts,'gamma')
            opts.gamma = 0.5;
        end
    case {lower('RBF')}
        if ~isfield(opts,'gamma')
            opts.gamma = 10;
        end
    case {lower('Polynomial')}      % (x'*y)^d
        if ~isfield(opts,'d')
            opts.d = 1;
        end
    case {lower('PolyPlus')}      % (x'*y+1)^d
        if ~isfield(opts,'d')
            opts.d = 1;
        end
    case {lower('tanh')}          
        if ~isfield(opts,'g')||~isfield(opts,'c')
            opts.g = 1;
            opts.c = 1;
        end
    otherwise
       error('KernelType does not exist!');
end
switch lower(opts.KernelType)       %把字母转为小写
    case {lower('Gaussian')}        %把字母转为小写     
        if isempty(Y)               %看Y是不是空矩阵，如果是则执行以下语句
            for i=1:N
               for j=i:N
%                    dist = sum(((X(:,i) - X(:,j)).^2));
                    dist = norm(X(:,i) - X(:,j)) .^2;
                    temp=exp(-opts.gamma*dist);
                    K(i,j)=temp;
                    if i~=j
                        K(j,i)=temp;
                    end;
                end
            end
        else
            for i=1:N
               for j=1:M
%                     dist = sum(((X(:,i) - Y(:,j)).^2));
                    dist =norm(X(:,i) - Y(:,j)).^2;
                    K(i,j)=exp(-opts.gamma*dist);                  
                end
            end
        end      
    case {lower('Polynomial')}    
        if isempty(Y)
            for i=1:N
                for j=i:N                   
                    temp=(X(:,i)'*X(:,j))^opts.d;
                    K(i,j)=temp;
                    if i~=j
                        K(j,i)=temp;
                    end;
                end
            end
        else
            for i=1:N
                for j=1:M                                      
                    K(i,j)=(X(:,i)'*Y(:,j))^opts.d;
                end
            end
        end      
    case {lower('PolyPlus')}    
        if isempty(Y)
            for i=1:N
                for j=i:N                   
                    temp=(X(:,i)'*X(:,j)+1)^opts.d;
                    K(i,j)=temp;
                    if i~=j
                        K(j,i)=temp;
                    end;
                end
            end
        else
            for i=1:N
                for j=1:M                                      
                    K(i,j)=(X(:,i)'*Y(:,j)+1)^opts.d;
                end
            end
        end
    case {lower('RBF')}      
        if isempty(Y)
            for i=1:N
                for j=i:N
%                     dist = sum(((X(:,i) - X(:,j)).^2));
                    dist = norm(X(:,i) - X(:,j)).^2;
%                     temp=exp(-0.5*dist/opts.gamma^2);
                    temp=exp(-10*dist/opts.gamma^2);
                    K(i,j)=temp;
                    if i~=j
                        K(j,i)=temp;
                    end;
                end
            end
        else
            for i=1:N
                for j=1:M
%                     dist = sum(((X(:,i) - Y(:,j)).^2));
                    dist = norm(X(:,i) - Y(:,j)).^2;
%                     K(i,j)=exp(-0.5*dist/opts.gamma^2);
                     K(i,j)=exp(-10*dist/opts.gamma^2);
                end
            end
        end      
    case {lower('tanh')}
         if isempty(Y)
            for i=1:N
                for j=i:N                   
                    temp=opts.g*X(:,i)*X(:,j)'+opts.c;
                    K(i,j)=temp;
                    if i~=j
                        K(j,i)=temp;
                    end
                end
            end
        else
            for i=1:N
                for j=1:M                                      
                    K(i,j)=opts.g*X(:,i)*Y(:,j)'+opts.c;
                end
            end 
        end      
    otherwise
        error('KernelType does not exist!');
end
