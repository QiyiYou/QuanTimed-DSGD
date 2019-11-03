%% NN parameters
input_layer_size  = 784;  % 28*28 Input Images 
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
w= eye (50) - L / (max(max(D))+1);
for i =1 :50
    wd(i,i) = w(i,i);
end

%% load data (MNIST or CIFAR-10)
% (features, labels) 

%% update iterations
mu=0; % mean computation time of each worker
T=0;% deadline time
lambda = 1;
num_labels = 10 ;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size,.5);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels,.5);
vmin = 10;
vmax = 90;

nn_params0 = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params0f = repmat(nn_params0,1,nworkers);


nn_params = nn_params0f;
t = zeros(nworkers, 1);
niterationsDGD = 700; 
alphadgd = 0.2;
niterationsDGDvec = niterations / 4; 
ttotal = zeros (2,niterationsDGD);
costDGD = zeros (2,niterationsDGD);
t2 = 1;

for batchsize = [20,50]
    nn_params= nn_params0f;
    for iteration = 1: niterationsDGD % DSGD
        for j= 1: nworkers % for workers 1,2,3,...,50
            
            gradnew(:,j) = nnCostFunctionDGD(nn_params(:,j), input_layer_size, hidden_layer_size, ...
                num_labels, features((j-1)*floor(10000/nworkers)+1:j*floor(10000/nworkers),:), labels((j-1)*floor(10000/nworkers)+1:j*floor(10000/nworkers)), lambda,T,mu,batchsize);
            t(j)=batchsize/(rand(1) * (vmax-vmin)+vmin);             
        end
 
        ttotal(t2,iteration) = max(t) + 3 ;
        nn_params = nn_params * w' - alphadgd * gradnew; % new theta
        costDGD (t2,iteration) = costf(mean(nn_params,2),features,labels,lambda,hidden_layer_size,input_layer_size,num_labels);
    end
    t2 = t2+1 ;
end

%% plot
semilogy (cumsum(ttotal(1,:)), costDGD(1,:),'linewidth',2);
hold on
semilogy (cumsum(ttotal(2,:)), costDGD(2,:),'linewidth',2)

