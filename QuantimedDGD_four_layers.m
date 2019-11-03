%% NN parameters
input_layer_size   = 1024;  % 32*32 Input Images 
hidden_layer1_size = 30;   
hidden_layer2_size = 20;   
hidden_layer3_size = 20;   
hidden_layer4_size = 25;   
nworkers = 50; 

%% generating communication graph
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
for i = 1:50
    wd(i,i) = w(i,i);
end


%% load data (MNIST or CIFAR-10)
% (features, labels) 

%% update iterations
mu = 0; % mean computation time of each worker
T = 0;% deadline time
lambda = 1;
num_labels = 10 ;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size,.5);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size,.5);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, hidden_layer3_size,.5);
initial_Theta4 = randInitializeWeights(hidden_layer3_size, hidden_layer4_size,1);
initial_Theta5 = randInitializeWeights(hidden_layer4_size, num_labels,2);

nn_params0 = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; initial_Theta4(:) ; initial_Theta5(:)];
nn_params0f = repmat(nn_params0,1,nworkers);
nn_params = nn_params0f;
gradnew = zeros (length(nn_params0),nworkers);


c1= 0.8; %Step Sizes
c2= 9;
vmin = 10; % minimum comupations speed
vmax = 90; % maximum computations speed

T = 50/50;
t1 = 1 ;

for niterations= 100:200:1100
   
    nn_params = nn_params0f;
    alpha = c1/ (niterations^(1/6)) ;
    epsilon = c2/ (niterations^(1/2)) ; 
    cost = zeros (niterations,1);
    for iteration = 1: niterations 
        for j= 1: nworkers % for workers 1,2,3,...,50
            gradnew(:,j) = nnCostFunction_four_layers(nn_params(:,j), input_layer_size, hidden_layer1_size,...
                hidden_layer2_size, hidden_layer3_size,hidden_layer4_size, ...
                num_labels, features((j-1) * floor(size(features,1)/nworkers)+1:j*floor(size(features,1)/nworkers),:), labels((j-1)*floor(size(features,1)/nworkers)+1:j*floor(size(Xnewnew,1)/nworkers)), lambda,T,vmin,vmax);
               
        end
        quantized = quantization (nn_params,5);
        nn_params = epsilon * (quantized) * (w'-wd) + epsilon * nn_params * wd + (1-epsilon) * nn_params - alpha * epsilon * gradnew; % new theta
        cost (iteration) = costf_four_layers (mean(nn_params,2), features, labels,lambda,input_layer_size, hidden_layer1_size,...
        hidden_layer2_size, hidden_layer3_size,hidden_layer4_size, num_labels);
    end
    finalcostDB (t1) = cost (niterations); 
    finaltimeDB (t1) = (T + 5/16 * 3) * niterations; 
    t1 = t1+1 ; 
end

