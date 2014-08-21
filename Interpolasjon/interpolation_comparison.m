%This script is intended to show the difference between several methods of
%approximating a given function by a polynomial.

%--------------------------------------
%   Setting global parameters:
%--------------------------------------

%Number of nodes:
N = 5;
%Nodes:
X = linspace(0,1,N+1);

%steepness of f:
s = 0.01;
f = @(x) exp(-(x-0.5).^2/s);

fvec=f(X);

%-------------------------
% Approximation by minimizing L_2-norm.
%       Monomial basis.
%-------------------------

%Creating matrix:
M = zeros(N+1);
for i=0:N
    for j=0:N
        M(i+1,j+1) = 1/(1+i+j);
    end
end

%Creating vector:
b = zeros(N+1,1);
for i=0:N
    b(i+1) = trapz(X, fvec.*X.^i);
end

coeffs = M\b;
    p = @(x) ones(length(x),1);
for i=1:N
    p = @(x) [p(x) x.^i];
end

x=linspace(0,1,50);
plot(x,p(x')*coeffs,'r')
hold on
plot(x,f(x),'b')
plot(X,zeros(N+1,1),'go')