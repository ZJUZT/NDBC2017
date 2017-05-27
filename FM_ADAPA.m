rng('default')

C = 1e-1;
eps = 0;
delta = 1e-1;
factor_num = 10;
iter_num = 1;

test = test_data;
num_train = size(train_data,1);
num_test = size(test,1);

MAE_ADAPA = zeros(iter_num, num_train);
MAEtest_ADAPA = zeros(iter_num);
% RMSE = zeros(1, num_train);

num_user = max(train_data(:,1));
num_item = max(train_data(:,2));

for i = 1:iter_num
    % shuffle
    re_idx = randperm(num_train);
    training = train_data(re_idx,:);
    
    U = rand(num_user, factor_num);
    V = rand(num_item, factor_num);
    
    U_ada = zeros(size(U,1),size(U,2));
    V_ada = zeros(size(V,1),size(V,2));
    
    for j = 1:num_train
        if mod(j,1e3)==0
            fprintf('%d iter---processing %dth sample\n', i, j);
        end
        a = training(j,1);
        b = training(j,2);
        r = training(j,3);
        r_hat = U(a,:)*(V(b,:)');

        if j == 1
            MAE_ADAPA(i,j) = abs(r-r_hat);
        else
            MAE_ADAPA(i,j) = (MAE_ADAPA(i,j - 1)*(j-1) + abs(r-r_hat)) / j;
        end
        
        loss = max(abs(r-r_hat)-eps,0);

        u_g = sign(r-r_hat)*(V(b,:));
        U_ada(a,:) = U_ada(a,:) + u_g.*u_g;
        v_g = sign(r-r_hat)*(U(a,:));
        V_ada(b,:) = V_ada(b,:) + v_g.*v_g;
        
        V_A = delta+(V_ada(b,:));
        U_A= delta+(U_ada(a,:));
        
        alpha_t3 = loss/(((V(b,:)./U_A*(V(b,:)')))+1/2/C);
        beta_t3 = loss/(((U(a,:)./V_A*U(a,:)'))+1/2/C);  

        U(a,:) = max(U(a,:) + sign(r-r_hat)*alpha_t3*V(b,:)./U_A,0);   
        V(b,:) = max(V(b,:) + sign(r-r_hat)*beta_t3*U(a,:)./V_A,0);   
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
    
    MAEtest_ADAPA(i) = err/num_test;
    
end

%% plot
% plot
plot(MAE_ADAPA,'DisplayName','SFM-AdaPA');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('MAE'); 
grid on; 
hold on;
