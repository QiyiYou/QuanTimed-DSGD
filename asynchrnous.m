%% NN parameters
input_layer_size  = 784;  % 32*32 Input Images 
hidden_layer_size = 50;   % hidden units
nworkers =50; 

%% generate communication graph
W0 = binornd(1,.4,[50,50]); %adjacency matrix is a bernoulli random matrix with p= 0.4
for i=1:50
    W0(i,i)=1;
end
for i = 1 : 50 
    for j = i: 50
        W0(i,j) = W0(j,i);
    end
end
D = zeros (50,50) ; 
for i = 1 : 50
    D(i,i) = sum(W0(i,:));
end
L = D-W0; 
wd = zeros (50,50);
w = eye (50) - L / (max(max(D))+1);
for i =1 :50
    wd(i,i) = w(i,i);
end

%% load data (MNIST or CIFAR-10)
% (features, labels) 

%% update iterations
mu = 0; % mean computation time of each worker
T = 0;% deadline time
lambda = 1;
num_labels = 10 ;
vmin = 10;
vmax = 90;
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size,.5);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels,.5);


nn_params0 = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params0f = repmat(nn_params0,1,nworkers);


nn_params = nn_params0f;
alphadgd= 0.2;
niterationsDGD = 900; 
t2 = 1;
Tcomm = 3;
batchsize = 30; 

time_slot = zeros(nworkers,niterationsDGD);
time_slot = batchsize ./ ( rand(nworkers,niterationsDGD) * (vmax-vmin)+vmin) + Tcomm;             
cum_time_slot = cumsum(time_slot,2) ;
sortedtime = sort(cum_time_slot(:)); 
workerindex = zeros(size(sortedtime));

for i = 1:length(sortedtime)
    workerindex(i) = rem(find(cum_time_slot==sortedtime(i))-1,nworkers) + 1; 
end

for iteration = 1: length(sortedtime)
    
    j = workerindex(iteration) ; 
    gradnew(:,j) = nnCostFunctionDGD(nn_params(:,j), input_layer_size, hidden_layer_size, ...
        num_labels, features((j-1)*floor(10000/nworkers)+1:j*floor(10000/nworkers),:), labels((j-1)*floor(10000/nworkers)+1:j*floor(10000/nworkers)), lambda,T,mu,batchsize);
    t(iteration) = sortedtime(iteration);             
 
    nn_params(:,j) = nn_params * w(j,:)' - alphadgd * gradnew(:,j); % new theta
    costDGD (iteration) = costf(mean(nn_params,2),features,labels,lambda,hidden_layer_size,input_layer_size,num_labels);
end

%% plot
plot(t, costDGD) ; 
