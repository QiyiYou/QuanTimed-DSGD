input_layer_size  = 784;  % 28*28 Input Images MNIST
hidden_layer_size = 50;   % hidden units
nworkers =50; 


W0=binornd(1,.4,[50,50]); %adjacency matrix is a bernoulli random matrix with p= 0.4
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
L= D-W0 ; 
wd = zeros (50,50);
w= eye (50) - L / (max(max(D))+1) ;
for i =1 :50
    wd(i,i) = w(i,i);
end

%displayData(Xnew(sel, :)); data load
mu=0; % mean computation time of each worker
T=0;% deadline time
lambda = 1; %regularizer
num_labels = 10 ;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size,.5);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels,.5);


nn_params0 = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params0f= repmat(nn_params0,1,nworkers);
nn_params= nn_params0f;
gradnew = zeros ((hidden_layer_size) *(input_layer_size+1) + (hidden_layer_size+1)* (num_labels),nworkers);


c1= 0.3; %Step Sizes
c2= 15;
vmin = 10; % minimum comupations speed
vmax = 90; % maximum computations speed
t2=1;
for T = [20/50 ,50/50]
t1 = 1 ;
for niterations= 100:200:1100
   
    nn_params = nn_params0f;

 alpha = c1/ (niterations^(1/6)) ;
 epsilon = c2/ (niterations^(1/2)) ; 
 cost = zeros (niterations,1);
for iteration = 1: niterations 
 for j= 1: nworkers % for workers 1,2,3,...,50
  gradnew(:,j) = nnCostFunction(nn_params(:,j), input_layer_size, hidden_layer_size, ...
                  num_labels, Xnewnew((j-1)*floor(10000/nworkers)+1:j*floor(10000/nworkers),:), labelstotal((j-1)*floor(10000/nworkers)+1:j*floor(10000/nworkers)), lambda,T,vmin,vmax);
               
 end
 quantized = quantization (nn_params,3) ;
 nn_params = epsilon * (quantized) * (w'-wd) + epsilon * nn_params * wd + (1-epsilon) * nn_params - alpha * epsilon * gradnew; % new theta
 cost (iteration) = costf(mean(nn_params,2), Xnewnew, labelstotal,lambda, hidden_layer_size, input_layer_size, num_labels);
end
finalcostDB (t2,t1) = cost (niterations) ; 
finaltimeDB (t2,t1) = (T + 3/16 * 3) * niterations ; 
t1=t1+1 ; 
end
t2=t2+1;
end
semilogy ([0,finaltimeDB(1,:)],[cost(1),finalcostDB(1,:)],'-*','linewidth',1);
hold on
semilogy ([0,finaltimeDB(2,:)],[cost(1),finalcostDB(2,:)],'-*','linewidth',1);
