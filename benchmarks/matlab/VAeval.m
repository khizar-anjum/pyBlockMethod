function [R0,R1] = VAeval(Z,Hes,varargin) % Vand.+Arnoldi basis construction
% Input:  Z = column vector of sample points
%         Hes = cell array of Hessenberg matrices
%         Pol = cell array of vectors of poles, if any
% Output: R0 = matrix of basis vectors for functions
%         R1 = matrix of basis vectors for derivatives
%
% From: Costa & Trefethen, "AAA-least squares rational approximation and
%       solution of Laplace problems", EMS Press, 2023.

M = length(Z); Pol = []; if nargin == 3, Pol = varargin{1}; end

% First construct the polynomial part of the basis
H = Hes{1}; Hes(1) = []; n = size(H,2);
Q = ones(M,1); D = zeros(M,1);
for k = 1:n
    hkk = H(k+1,k);
    Q(:,k+1) = ( Z.*Q(:,k) - Q(:,1:k)*H(1:k,k)           )/hkk;
    D(:,k+1) = ( Z.*D(:,k) - D(:,1:k)*H(1:k,k) + Q(:,k)  )/hkk;
end
R0 = Q; R1 = D;

% Next construct the pole parts of the basis, if any
while ~isempty(Pol)
    pol = Pol{1}; Pol(1) = [];
    H = Hes{1}; Hes(1) = []; np = length(pol); Q = ones(M,1); D = zeros(M,1);
    for k = 1:np
        Zpki = 1./(Z-pol(k)); hkk = H(k+1,k);
        Q(:,k+1) = ( Q(:,k).*Zpki - Q(:,1:k)*H(1:k,k)                     )/hkk;
        D(:,k+1) = ( D(:,k).*Zpki - D(:,1:k)*H(1:k,k) - Q(:,k).*Zpki.^2  )/hkk;
    end
    R0 = [R0 Q(:,2:end)]; R1 = [R1 D(:,2:end)];
end
end
