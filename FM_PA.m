rng('default')

C = 1e-2;
eps = 0;
factor_num = 10;
iter_num = 1;

test = test_data;
num_train = size(train_data,1);
num_test = size(test,1);

MAE_PA = zeros(iter_num, num_train);
MAEtest_PA = zeros(iter_num);
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
            MAE_PA(i,j) = abs(r-r_hat);
        else
            MAE_PA(i,j) = (MAE_PA(i,j - 1)*(j-1) + abs(r-r_hat)) / j;
        end
        
        loss = max(abs(r-r_hat)-eps,0);

        alpha_t = loss/(V(b,:)*V(b,:)'+1/2/C);
        alpha_t1 = min(C,loss/(V(b,:)*V(b,:)'));
        beta_t = loss/(U(a,:)*U(a,:)'+1/2/C);
        beta_t1 = min(C,loss/(U(a,:)*U(a,:)'));
        
        U(a,:) = U(a,:) + sign(r-r_hat)*alpha_t*V(b,:);
        V(b,:) = V(b,:) + sign(r-r_hat)*beta_t*U(a,:);   
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
    
    MAEtest_PA(i) = err/num_test;
    
end

%% plot
% plot
plot(MAE_PA,'DisplayName','SFM-PA');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('MAE'); 
grid on; 
hold on;
