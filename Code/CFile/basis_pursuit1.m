function [ x, history, iter ] = basis_pursuit1( A, b, rho1, rho2, alpha, lambda, theta, regtype )
% basis_pursuit  Solve basis pursuit via ADMM

%t_start = tic;
%Global constants and defaults

QUIET    = 0;
MAX_ITER = 300;
ABSTOL   = 1e-5;
RELTOL   = 1e-5;
%Data preprocessing
relax_alpha=1.8;

[d, n] = size(A);
%ADMM solver



x = randn(d,1);
%disp(x)
z=x;
%z = zeros(d,1);
xsi=randn(n,1);
s=xsi;
u1=z;
u2=s;
%s=zeros(n,1);
%u1=zeros(n,1);
%u2=zeros(d,1);
%vector_onen=ones(n,1);
%vector_oned=ones(d,1);
%u = zeros(n,1);

history.obj = zeros(MAX_ITER+1,1);
history.time = history.obj;

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm1', 'eps pri1', 's norm1', 'eps dual1','r norm2', 'eps pri2', 's norm2', 'eps dual2', 'objective');
end


% precompute static variables for x-update (projection on to Ax=b)
Y=sparse(1:n,1:n,b,n,n);
H=Y*A';

HtH=H'*H;

history.objval(1) = alpha*sum(max(0,1-H*x))+funRegC(z,d,1,theta,regtype);
%disp(history.fun(1));
history.time(1)=0;

for k = 1:MAX_ITER
    tic;
    
    % x-update
   d1=(rho1*eye(d)+rho2*HtH);
   d2=rho1*(z-u1)+rho2*H'*(s+1-xsi-u2);
   x=(d1\d2);
  
    % z-update with relaxation   
    zold = z;
    z= proximalRegC(x + u1, d, lambda/rho1, theta, regtype);
 
    % xsi updation
    xsi=-alpha/rho2-H*x+1+s-u2;
    xsi= max(0, xsi);
  
    %s updation
    sold=s;
    s=H*x+xsi-1+u2;
    s= max(0, s);

    % dual variable updation
    u1=u1+(x-z);
    u2=u2+(H*x+xsi-1-s);
   

    history.time(k+1) = history.time(k) + toc;
    % diagnostics, reporting, termination checks
    history.objval(k+1)=alpha*sum(max(0,1-H*x))+funRegC(z,d,1,theta,regtype);
    %disp(history.objval(k));

    history.r_norm1(k)= norm(x - z);
    history.s_norm1(k)= norm(-rho1*(z - zold));
    
    history.r_norm2(k)  = norm(xsi-s+H*x-1);
    history.s_norm2(k)  = norm(-rho2*(s - sold));
    
    history.eps_pri1(k) = sqrt(d)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual1(k)= sqrt(d)*ABSTOL + RELTOL*norm(rho1*u1);

    temp=max(norm(xsi), norm(-s));
    history.eps_pri2(k) = sqrt(n)*ABSTOL + RELTOL*max(temp, norm(H*x-1));
    history.eps_dual2(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho2*u2);
    
   

    if ~QUIET
        fprintf('%3d\t %10.4f\t%10.4f\t %10.4f\t%10.4f\t %10.2f\t%10.4f\t %10.4f\t%10.4f\t %10.2f\n', k, ...
            history.r_norm1(k), history.eps_pri1(k), ...
            history.s_norm1(k), history.eps_dual1(k), ...
             history.r_norm2(k), history.eps_pri2(k), ...
            history.s_norm2(k), history.eps_dual2(k),history.objval(k));
    end

    
    if (history.r_norm1(k) < history.eps_pri1(k) && ...                                                                                                                                                                                                                                                  
       history.s_norm1(k) < history.eps_dual1(k) && ...
       history.r_norm2(k) < history.eps_pri2(k) && ...
       history.s_norm2(k) < history.eps_dual2(k))
         disp('end');
         break;
    end
end

%if ~QUIET
  %  toc(t_start);
%end
history.objval = history.objval(1: min(MAX_ITER,k)+1);
history.time = history.time(1: min(MAX_ITER,k)+1);

iter = k;
end







