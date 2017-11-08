clear;clc;
close all;

%% load the .mat train data 
%trainData = load ('mushrooms.mat');
%Data = load ('w8a.mat');
% rho1=0.001 rho2=0.001 alpha=10
%Data = load ('rcv1_train.mat');
% alpha=0.04
%Data = load ('splice.mat');
% rho1=0.001 rho2=0.001 alpha=0.1
%Data = load ('svmguide3.mat');
%Data = load ('svmguide3.mat');

% A:d*n  b:n*1
%A = trainData.samples;
%b = trainData.labels;

%% load the .txt train data 
%trainData = load ('mushrooms.mat');
%[label, instance]=libsvmread('svmguide1.txt');
%Data = load ('w8a.mat');
% rho1=0.001 rho2=0.001 alpha=10
%Data = load ('rcv1_train.mat');
% alpha=0.04
%Data = load ('splice.mat');
% rho1=0.001 rho2=0.001 alpha=0.08
%Data = load ('svmguide3.mat');
%Data = load ('svmguide3.mat');
%[label, instance]=libsvmread('');
[train_label, train_instance]=libsvmread('.\dataset\madelon');
disp(size(train_label))
disp(size(train_instance))
% Convert the label to correct {-1, 1} class if it is required
ConvertLable=1;
if(ConvertLable==1)
    for i = 1:length(train_label),
        if(train_label(i,1)~=1)
            train_label(i,1) = -1;
        end
    end
end
%X=instance;
%y=label; 
% A:d*n  b:n*1
A = train_instance';
b = train_label;


%% ================================ Liblinear train =================================
% train the data 
 t_start = tic;
 model = train(train_label, train_instance, '-s 5 -c 1');
 toc(t_start);
 fprintf('The elaspsed time of liblinear:%.2f s\n',t_start);

% pause();
%% =============================== Train of ADMM solver=================================
rho1=0.4;
rho2=1.4;
alpha=0.18; % fixed
% input parameters
%lambda = 1e-3*abs(randn);
lambda=0.001*rho1;
%theta = 1e-2*lambda*abs(randn);
theta=380.00;
%theta=0.01*lambda;
%theta = inf;
regtype=2;
%Solve problem
%basis_pursuit(A, b, rho1,rho2, alpha,lambda,regtype)

fprintf('Training the Data ...\n');
%disp(size(x));
%disp(size(A));
%pause();
%alpha_text=[0.001,0.1,0.2,0.5,1,2,4,8,10]';

[x, history, iter] = basis_pursuit1(A, b, rho1, rho2, alpha, lambda, theta, regtype);
%[x, history, iter] = nonconvexSolver( A, b, alpha, rho1, lambda, theta, regtype);
disp(x);
 



%display(x);

%K = length(history.objval);


%% ================================= plot the result ====================================

figure;
subplot(2,1,1);
semilogy(1:iter+1, history.objval(1:iter+1), 'r-', 'MarkerSize', 10, 'LineWidth', 2)
%plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('obj'); xlabel('iter (k)');

subplot(2,1,2);
semilogy(history.time(1:iter+1), history.objval(1:iter+1), 'r-', 'MarkerSize', 10, 'LineWidth', 2)
ylabel('Objective function value (logscaled)'); xlabel('CPU time (seconds)');
%g = figure;
%subplot(2,1,1);
%semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
   % 1:K, history.eps_pri, 'k--',  'LineWidth', 2);
%ylabel('||r||_2');

%subplot(2,1,2);
%semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    %1:K, history.eps_dual, 'k--', 'LineWidth', 2);
%ylabel('||s||_2'); xlabel('iter (k)');

clear train_label;
clear train_instance;


%% ================================load the test data=========================================
% load the test data 
% rcv1_test.binary
[test_label, test_instance]=libsvmread('.\dataset\madelon.t');
ConvertLable=1;
if(ConvertLable==1)
    for i = 1:length(test_label),
        if(test_label(i,1)~=1)
            test_label(i,1) = -1;
        end
    end
end

%% ================================predict for liblinear=========================================
% train the data 
[predicted_label, accuracy, decision_values] = predict(test_label, test_instance, model);
%disp(accuracy);
%fprintf('The prediction accuracy of liblinear is %.2f\n\n',accuracy);
%% ================================predict for admm solver=======================================
% load the .txt prediction data 

A = test_instance;
b = test_label;
%disp(size(A));
%disp(size(x));
result=A*x;
num=0;

for i = 1:length(b)
    if(result(i,1)>0 && b(i,1)==1)
        num = num + 1;
    elseif(result(i,1)<0 && b(i,1)==-1)
        num = num + 1;
    end
end

fprintf('The test error of ADMM solver is:%.2f%%\n', (length(b)-num)/length(b)*100);



clear test_label;
clear test_instance;

