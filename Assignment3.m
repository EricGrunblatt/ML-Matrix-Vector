% Assignment 3: Perceptron
% Eric Grunblatt

%%% PART 1: LINEAR SEPARABLE DATA %%%
% Linear Separable Data Set
X = dlmread('X_LinearSeparable.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y_LinearSeparable.txt'); %Y: 1*N
% Get the rows, columns, set the weights and threshold
numRows = size(X,1);
numCols = size(X,2);
weights = zeros(1, numRows); % Initialize the weights
threshold = 0;
% PLA Algorithm
errors = 0;
total = 0;
currentErrors = 1;
while(currentErrors ~= 0)
    currentErrors = 0;
    random = randperm(numCols);
    for a=1:numCols % Iterating through each vector
        i = random(a);
        currentNum = weights * X(:,i);
        % Identifying whether there is an error or not, if so, correct it and
        % add to the total number of errors
        total = total + 1;
        if(sign(currentNum) ~= sign(Y(i)) || currentNum == threshold)
            currentErrors = currentErrors + 1;
            weights = weights + transpose(Y(i)*X(:,i)); % If error, add to weights
        end     
    end
    errors = errors + currentErrors;   
end
% Plotting Points
figure;
for i=1:numCols
    if(Y(1,i) == 1) % Y is shown as 1, then it is greater than threshold
        plot(X(2,i), X(3,i), 'o', 'color', 'blue');
        hold on
    else % Y is shown as -1, and is less than the threshold
        plot(X(2,i), X(3,i), 'x', 'color', 'red');
        hold on
    end
end
% Creating/Plotting boundary line
slope = -weights(2)/weights(3);  
%y =mx+c, m is slope and c is intercept(0)
x = [min(X(2,:)),max(X(2,:))];
y = (slope*x);
line(x, y);
hold off
% Report error rate
fprintf('Linear Data Error Rate: %f\n', (errors/total));


%%% PART 2: NOISY DATA %%%
% Noisy Data Set
X = dlmread('X_NonLinearSeparable.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y_NonLinearSeparable.txt'); %Y: 1*N
% Get the rows, columns, set the weights and threshold
numRows = size(X,1);
numCols = size(X,2);
weights = zeros(1,numRows); % Initialize the weights
threshold = 0;
% Pocket Algorithm
errors = 0;
total = 0;
w_star = weights;
w_star_errors = 1;
enoughIterations = 0;
while(enoughIterations < 200)
    random = randperm(numCols);
    for b=1:numCols % Iterating through each vector
        i = random(b);
        currentNum = weights * X(:,i);
        % Identifying whether there is an error or not, if so, correct it and
        % add to the total number of errors
        total = total + 1;
        if(sign(currentNum) ~= sign(Y(i)) || currentNum == threshold)
            errors = errors + 1;
            enoughIterations = enoughIterations + 1;
            weights = weights + transpose(Y(i)*X(:,i)); % If error, add to weights
            new_weights_errors = 0;
            % Check if w* or w(t+1) has more errors
            for c=1:numCols
                index = random(c);
                weightNum = weights * X(:,index);
                starNum = w_star * X(:,index);
                if(sign(starNum) ~= sign(Y(index)) || starNum == threshold)
                    w_star_errors = w_star_errors + 1;
                end
                if(sign(weightNum) ~= sign(Y(index)) || weightNum == threshold)
                    new_weights_errors = new_weights_errors + 1;
                end
            end
            % If w* has more errors, then change w* to w(t+1)
            if(new_weights_errors < w_star_errors) % If there is no error, then break out of the loop
                w_star_errors = new_weights_errors;
                w_star = weights;
            end  
        end     
    end
end
% Plotting Points
figure;
for i=1:numCols
    if(Y(1,i) == 1) % Y is shown as 1, then it is greater than threshold
        plot(X(2,i), X(3,i), 'o', 'color', 'blue');
        hold on
    else % Y is shown as -1, and is less than the threshold
        plot(X(2,i), X(3,i), 'x', 'color', 'red');
        hold on
    end
end
% Creating/Plotting boundary line
slope = -w_star(2)/w_star(3);  
%y =mx+c, m is slope and c is intercept(0)
x = [min(X(2,:)),max(X(2,:))];
y = (slope*x);
line(x, y);
hold off
% Report error rate
fprintf('Non-Linear Data Error Rate: %f\n', (errors/total));


%%% PART 3: HANDCRAFTED FEATURES %%%
% Handcrafted Features Train
X = dlmread('X_Digits_HandcraftedFeature_Train.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y_Digits_HandcraftedFeature_Train.txt'); %Y: 1*N
% Get the rows, columns, set the weights and threshold
numCols = size(X,2);
weights = [0.1, 0, 0]; % Only works if the first number is not a whole number or 0
threshold = 0;
% Pocket Algorithm
errors = 0;
total = 0;
w_star = weights;
w_star_errors = 1;
enoughIterations = 0;
while(enoughIterations < 200)
    random = randperm(numCols);
    for b=1:numCols % Iterating through each vector
        i = random(b);
        currentNum = weights * X(:,i);
        % Identifying whether there is an error or not, if so, correct it and
        % add to the total number of errors
        total = total + 1;
        enoughIterations = enoughIterations + 1;
        if(sign(currentNum) ~= sign(Y(i)) || currentNum == threshold)
            errors = errors + 1;
            weights = weights + transpose(Y(i)*X(:,i)); % If error, add to weights
            new_weights_errors = 0;
            % Check if w* or w(t+1) has more errors
            for c=1:numCols
                index = random(c);
                weightNum = weights * X(:,index);
                starNum = w_star * X(:,index);
                if(sign(starNum) ~= sign(Y(index)) || starNum == threshold)
                    w_star_errors = w_star_errors + 1;
                end
                if(sign(weightNum) ~= sign(Y(index)) || weightNum == threshold)
                    new_weights_errors = new_weights_errors + 1;
                end
            end
            % If w* has more errors, then change w* to w(t+1)
            if(new_weights_errors < w_star_errors) % If there is no error, then break out of the loop
                w_star_errors = new_weights_errors;
                w_star = weights;
            end  
        end     
    end
end
% Handcrafted Features Test
X = dlmread('X_Digits_HandcraftedFeature_Test.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y_Digits_HandcraftedFeature_Test.txt'); %Y: 1*N
numCols = size(X,2);
errors = 0;
total = 0;
% Error Rate
random = randperm(numCols);
for a=1:numCols
    i = random(a);
    total = total + 1;
    currentNum = weights * X(:,i);
    if(sign(currentNum) ~= sign(Y(i)) || currentNum == threshold)
        errors = errors + 1;
    end
end
% Report error rate
fprintf('Handcrafted Feature Data Error Rate: %f\n', (errors/total));

% Plotting Points
figure;
for i=1:numCols
    if(Y(1,i) == 1) % Y is shown as 1, then it is greater than threshold
        plot(X(2,i), X(3,i), 'o', 'color', 'blue');
        hold on
    else % Y is shown as -1, and is less than the threshold
        plot(X(2,i), X(3,i), 'x', 'color', 'red');
        hold on
    end
end
% Creating/Plotting boundary line
slope = -w_star(2)/w_star(3);  
intercept = -w_star(1)/w_star(3);
%y =mx+c, m is slope and c is intercept
x = [min(X(2,:)),max(X(2,:))];
y = (slope*x) + intercept;
line(x, y);
hold off


%%% PART 4: RAW PIXEL FEATURES %%%
% Raw Pixel Features
X = dlmread('X_Digits_RawFeature_Train.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y_Digits_RawFeature_Train.txt'); %Y: 1*N
% Get the rows, columns, set the weights and threshold
numRows = size(X,1);
numCols = size(X,2);
weights = zeros(1,numRows); % Only works if the first number is not a whole number or 0
threshold = 0;
% Pocket Algorithm
errors = 0;
total = 0;
w_star = weights;
w_star_errors = 1;
enoughIterations = 0;
while(enoughIterations < 100)
    random = randperm(numCols);
    for b=1:numCols % Iterating through each vector
        i = random(b);
        currentNum = weights * X(:,i);
        % Identifying whether there is an error or not, if so, correct it and
        % add to the total number of errors
        total = total + 1;
        enoughIterations = enoughIterations + 1;
        if(sign(currentNum) ~= sign(Y(i)) || currentNum == threshold)
            errors = errors + 1;
            weights = weights + transpose(Y(i)*X(:,i)); % If error, add to weights
            new_weights_errors = 0;
            % Check if w* or w(t+1) has more errors
            for c=1:numCols
                index = random(c);
                weightNum = weights * X(:,index);
                starNum = w_star * X(:,index);
                if(sign(starNum) ~= sign(Y(index)) || starNum == threshold)
                    w_star_errors = w_star_errors + 1;
                end
                if(sign(weightNum) ~= sign(Y(index)) || weightNum == threshold)
                    new_weights_errors = new_weights_errors + 1;
                end
            end
            % If w* has more errors, then change w* to w(t+1)
            if(new_weights_errors < w_star_errors) % If there is no error, then break out of the loop
                w_star_errors = new_weights_errors;
                w_star = weights;
            end  
        end     
    end
end
% Handcrafted Features Test
X = dlmread('X_Digits_RawFeature_Test.txt'); %X: (1+d)*N; d=2 in this dataset for visualization; N=20 in this dataset
Y = dlmread('Y_Digits_RawFeature_Test.txt'); %Y: 1*N
numCols = size(X,2);
errors = 0;
total = 0;
% Error Rate
random = randperm(numCols);
for a=1:numCols
    i = random(a);
    total = total + 1;
    currentNum = weights * X(:,i);
    if(sign(currentNum) ~= sign(Y(i)) || currentNum == threshold)
         errors = errors + 1;
    end
end
% Report error rate
fprintf('Raw Feature Data Error Rate: %f\n', (errors/total));
