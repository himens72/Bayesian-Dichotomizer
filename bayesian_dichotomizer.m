% loading the samples (matrix file)
load samples.mat;
[n,m] = size(samples);
priorProb = [0.5 ,0.5, 0];

% Queastions a and b%
error = 0;
x1W = [samples(:,1), samples(1:10, 1+3), samples(:, 1+3*2)];
p = priorProb(1,:);
for i=1:10
    x0 = samples(i, 1);
    c = dichotomizer(x0, x1W, p);
    if c ~= 1
        error = error + 1;
    end
end

% for class 2 sample
for i=1:10
    x0 = samples(i, 4);
    c = dichotomizer(x0, x1W, p);
    if c ~= 2
        error = error + 1;
    end
end

% calculating the error rate
error_rate = error/20;
fprintf("Question B is: %f\n", error_rate);

% Use the Bhattacharyya bound to bound the error you will get on novel patterns drawn from the distributions.
mu1 = mean(samples(:,1:3)).';
mu2 = mean(samples(:,4:6)).';
sigma1 = cov(samples(:,1:3));
sigma2 = cov(samples(:,4:6));
k = (1/8) * (((mu2 - mu1)).') * inv((sigma1 + sigma2)/2) * (mu1 - mu2) + 0.5*log((det(((sigma1 + sigma2)/2))) / sqrt(det(sigma1) * det(sigma2)));
p_error = sqrt(0.25)*exp(-k);
fprintf("Question C : %f\n", p_error);

% for 2 featuers
error = 0;
x12W = [samples(:,1:2), samples(:, 1+3:2+3), samples(:, 1+3*2:2+3*2)];
for i=1:10
    x0 = samples(i, 1:2);
    c = dichotomizer(x0, x12W, p);
    if c ~= 1
        error = error + 1;
    end
end

% for class 2 sample
for i=1:10
    x0 = samples(i, 4:5);
    c = dichotomizer(x0, x12W, p);
    if c ~= 2
        error = error + 1;
    end
end

% calculating error rate
error_rate = error/20;
fprintf("Question D is: %f\n", error_rate);

% for 2 features 2 & 3
error = 0;
x23W = [samples(:,2:3), samples(:, 2+3:3+3), samples(:, 2+3*2:3+3*2)];
for i=1:10
    x0 = samples(i, 2:3);
    c = dichotomizer(x0, x23W, p);
    if c ~= 1
        error = error + 1;
    end
end

% for 2 features 1 & 3
error = 0;
x13W = [samples(:,1),samples(:,3), samples(:,4), samples(:,6), samples(:,7),samples(:, 9)];
for i=1:10
    x0 = [samples(i, 1), samples(i, 3)];
    c = dichotomizer(x0, x13W, p);
    if c ~= 1
        error = error + 1;
    end
end

% for 3 features
error = 0;
x123W = [samples(:,1:3), samples(:, 1+3:3+3), samples(:, 1+3*2:3+3*2)];
for i=1:10
    x0 = samples(i, 1:3);
    c = dichotomizer(x0, x123W, p);
    if c ~= 1
        error = error + 1;
    end
end

% for class 2 sample
for i=1:10
    x0 = samples(i, 4:6);
    c = dichotomizer(x0, x123W, p);
    if c ~= 2
        error = error + 1;
    end
end

% calculating error rate
error_rate = error/20;
fprintf("Question E: %f\n", error_rate);


function g = getData (x0, XW, priorProb)
    mu = mean(XW);
    sigma = cov(XW);
    multiVariate = mvnpdf(x0 , mu, sigma);
    g = log(multiVariate) + log(priorProb);
end

% function defination for dichotomizer
function  class = dichotomizer(x, XW, priorProb)
    [~, v] = size(XW);
    v = v/3-1;
    g1 = getData (x, XW(1:10, 1:1+v), priorProb(1));
    g2 = getData (x, XW(1:10, 2+v:2+2*v), priorProb(2));
    g3 = getData (x, XW(1:10, 3+2*v:3+3*v), priorProb(3));
    if max([g1,g2,g3]) == g1
        class = 1;
    elseif max([g1,g2,g3]) == g2
        class = 2;
    else
        class = 3;
    end
end

%Refernces
%https://www.mathworks.com/help/matlab/ref/cov.html
%https://www.mathworks.com/matlabcentral/answers/25283-how-to-calculate-with-sigma-notation-in-matlab
%https://www.mathworks.com/help/matlab/ref/load.html
%https://www.mathworks.com/help/matlab/matlab_env/save-load-and-delete-workspace-variables.html
%https://www.mathworks.com/help/matlab/ref/function.html
%https://www.mathworks.com/help/stats/examples/classification.html
%https://www.mathworks.com/help/stats/examples/classification.html#d117e4450
%https://www.mathworks.com/discovery/pattern-recognition.html