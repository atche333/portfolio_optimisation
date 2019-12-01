clear;
stocks = xlsread('temp.xls','A1:AC4964');
 
prod1 = stocks(:, 1);
prod2 = stocks(:, 2);
prod3 = stocks(:, 3 );
 
prices = [prod1,prod2,prod3];
mean = mean(prices);  % mean of the assets 
cov_mat = cov(prices);  %covariance matrix 
exp_ret = 11;  %setup for optimisation
fun = @(x) x * cov_mat * x';  % function is in the form of a QP problem 
x0 = [0.34,0.33,0.33];  %initial point
f=zeros(size(x0)); 
A = []; 
b = [];
Aeq = [1,1,1;mean];
beq = [1;exp_ret];
lb = [0,0,0];
ub = [0.5,0.5,0.5];
[ans3,ans4] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
[weights,varP,exitflag,output,lambdas]=quadprog(cov_mat,f,A,b,Aeq,beq,lb,ub);  % weights = proportion of assets in portfolio
varP = weights' * cov_mat * weights; %final variance at optimal weights
disp(cov_mat);
H = cov_mat;
Ain = A;
bin = b;
x = weights;
ans1 = computeKKTErrorForQPLP1(H,f,Ain,bin,Aeq,beq,lb,ub,lambdas,x);
disp(ans1);
function kktError = computeKKTErrorForQPLP1(H,f,Ain,bin,Aeq,beq,lb,ub,lambdas,x)
%computes first order optimality of a QP Problem from the KKT conditions.
 
if isempty(x)
kktError = [];
return;
end
nvars = length(x);
f = f(:);
x = x(:);
if isempty(Ain), Ain = zeros(0,nvars); end
if isempty(Aeq), Aeq = zeros(0,nvars); end
mIneq = size(Ain,1);
mEq = size(Aeq,1);
if isempty(lambdas.lower)
lambdas.lower = zeros(nvars,1);
end
if isempty(lambdas.upper)
lambdas.upper = zeros(nvars,1);
end
if isempty(lambdas.ineqlin)
lambdas.ineqlin = zeros(mIneq,1);
end
if isempty(lambdas.eqlin)
lambdas.eqlin = zeros(mEq,1);
end
if isempty(H)
grad = f;
else
grad = H*x + f;
end
 
normgradLag = norm(grad + Aeq'*lambdas.eqlin + Ain'*lambdas.ineqlin - lambdas.lower +lambdas.upper, Inf);
 
finiteLB = ~isinf(lb); lenLB = nnz(finiteLB);
finiteUB = ~isinf(ub); lenUB = nnz(finiteUB);
 
Comp = zeros(mIneq+lenLB+lenUB,1);
if mIneq > 0
Comp(1:mIneq) = lambdas.ineqlin.*(Ain*x-bin);
end
if lenLB > 0
Comp(mIneq+1:mIneq+lenLB) = (lb(finiteLB)' - x(finiteLB)).*(lambdas.lower(finiteLB));
end
if lenUB > 0
Comp(mIneq+lenLB+1:end) = (x(finiteUB) - ub(finiteUB)').*lambdas.upper(finiteUB);
end
normComp = norm(Comp,Inf);
 
kktError = max(normgradLag, normComp);
end
 
 
