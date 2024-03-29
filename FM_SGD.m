rng('default')

learning_rate = 1e-2;
reg = 1e-4;
factor_num = 10;
iter_num = 1;

test = test_data;
num_train = size(train_data,1);
num_test = size(test,1);

MAE_SGD = zeros(iter_num, num_train);
MAEtest_SGD = zeros(iter_num);
% RMSE = zeros(1, num_train);

num_user = max(train_data(:,1));
num_item = max(train_data(:,2));

for i = 1:iter_num
    % shuffle
    re_idx = randperm(num_train);
    training = train_data(re_idx,:);
    
    U = rand(num_user, factor_num);
    V = rand(num_item, factor_num);
    
    for j = 1:num_train
        if mod(j,1e3)==0
            fprintf('%d iter---processing %dth sample\n', i, j);
        end
        a = training(j,1);
        b = training(j,2);
        r = training(j,3);
        r_hat = U(a,:)*(V(b,:)');

        if j == 1
            MAE_SGD(i,j) = abs(r-r_hat);
        else
            MAE_SGD(i,j) = (MAE_SGD(i,j - 1)*(j-1) + abs(r-r_hat)) / j;
        end

        U(a,:) = U(a,:) - learning_rate * ((r_hat - r) * V(b,:) + 2 * reg * U(a,:)) ;
        V(b,:) = V(b,:) - learning_rate * ((r_hat - r) * U(a,:) + 2 * reg * V(b,:));   
    end
    
    tic;
    err = 0;
    for j = 1:size(test,1)
        a = test(j,1);
        b = test(j,2);
        r = test(j,3);
        r_hat = U(a,:)*(V(b,:)');
        err = err + abs(r-r_hat);
    end
    toc;
    
    MAEtest_SGD(i) = err/num_test;
    
end

%% plot
% plot
plot(MAE_SGD,'DisplayName','SFM-SGD');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('MAE'); 
grid on; 
hold on;
