%% DSGD fourlayerNN
input_layer_size   = 1024;  % 32*32 Input Images 
hidden_layer1_size = 30;   
hidden_layer2_size = 20;   
hidden_layer3_size = 20;   
hidden_layer4_size = 25;   
nworkers = 50; 


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

nn_params0 = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; initial_Theta4(:) ; initial_Theta5(:)];
nn_params0f = repmat(nn_params0,1,nworkers);
nn_params = nn_params0f;
gradnew = zeros (length(nn_params0),nworkers);


alphadgd = 0.08 ;
vmin = 10; % minimum comupations speed
vmax = 90; % maximum computations speed

batchsize = 50 ;

niterations = 800 ;
   
    nn_params = nn_params0f;

 costDGD = zeros (niterations,1);
 ttotal = zeros (niterations,1);
for iteration = 1: niterations 
 for j= 1: nworkers % for workers 1,2,3,...,50
  gradnew(:,j) = nnCostFunctionDGD_four_layers(nn_params(:,j), input_layer_size, hidden_layer1_size,...
                  hidden_layer2_size, hidden_layer3_size,hidden_layer4_size, ...
                  num_labels, Xnewnew((j-1) * floor(size(Xnewnew,1)/nworkers)+1:j*floor(size(Xnewnew,1)/nworkers),:), labelstotal((j-1)*floor(size(Xnewnew,1)/nworkers)+1:j*floor(size(Xnewnew,1)/nworkers)), lambda,T,batchsize);
       t(j)=batchsize/(rand(1) * (vmax-vmin)+vmin);                        
 end
ttotal(iteration) = max(t) + 3 ;
 quantized = quantization (nn_params,3) ;
 nn_params = nn_params * w' - alphadgd * gradnew ; % new theta
 costDGD (iteration) = costf_four_layers (mean(nn_params,2), Xnewnew, labelstotal,lambda,input_layer_size, hidden_layer1_size,...
 hidden_layer2_size, hidden_layer3_size,hidden_layer4_size, num_labels);
end
