function grad = nnCostFunctionDGD_four_layers(nn_params, ...
                                   il, ...
                                   hl1, ...
                                   hl2, ...
                                   hl3, ...
                                   hl4, ...
                                   num_labels, ...
                                   X, y, lambda,td,batchsize)

Theta1 = reshape(nn_params(1:hl1 * (il + 1)), ...
                 hl1, (il + 1));
i1 = hl1 * (il + 1) ; 
Theta2 = reshape(nn_params(i1+1:i1+hl2*(hl1+1)), ...
                 hl2, (hl1 + 1));
i2 = i1+hl2*(hl1+1) ;              
Theta3 = reshape(nn_params(i2+1:i2+hl3 * (hl2 + 1)), ...
                 hl3, (hl2 + 1));
i3 = i2+hl3 * (hl2 + 1) ;             
Theta4 = reshape(nn_params(i3+1:i3+hl4 * (hl3 + 1)), ...
                 hl4, (hl3 + 1));
i4 = i3+hl4 * (hl3 + 1) ;
Theta5 = reshape(nn_params(i4+1:i4+num_labels * (hl4 + 1)), ...
                 num_labels, (hl4 + 1));              
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));
Theta5_grad = zeros(size(Theta5));
    
r= randi(size(X,1),batchsize,1);
Xn= X(r,:);
yn= y(r,:);
m = size(Xn, 1);
X_copy=Xn;
Xn = [ones(m, 1) Xn];

z1 = Xn * Theta1' ;
a1 = sigmoid(z1);% first layer

a1 = [ones(size(a1,1),1) a1];

z2 = (a1 * Theta2'); 
a2 = sigmoid(z2)   ;
a2 = [ones(size(a2,1),1) a2];


z3 = (a2 * Theta3'); 
a3 = sigmoid(z3)   ;
a3 = [ones(size(a3,1),1) a3];

z4 = (a3 * Theta4'); % fourth hidden layer
a4 = sigmoid(z4)   ;
a4 = [ones(size(a4,1),1) a4];

h_theta=sigmoid(a4 * Theta5'); % computing htheta (x)

Y=zeros(m,num_labels);

% generating Matrix Y from Vector y:
for ic=1:m
Y(ic,yn(ic))=1;
end

z1 = [ones(size(z1,1),1) z1];
z2 = [ones(size(z2,1),1) z2];
z3 = [ones(size(z3,1),1) z3];
z4 = [ones(size(z4,1),1) z4];

Delta5 = zeros(size(Theta5,1),size(Theta4,1)+1);
Delta4 = zeros(size(Theta4,1),size(Theta3,1)+1);
Delta3 = zeros(size(Theta3,1),size(Theta2,1)+1);
Delta2 = zeros(size(Theta2,1),size(Theta1,1)+1);
Delta1 = zeros(size(Theta1,1),size(X_copy,2)+1);


for icc=1:m
 delta6 = transpose(h_theta(icc,:) - Y(icc,:));
 delta5 = (Theta5' * delta6 ) .* transpose(sigmoidGradient(z4(icc,:)));
 delta5 = delta5(2:end);
 delta4 = (Theta4' * delta5 ) .* transpose(sigmoidGradient(z3(icc,:)));
 delta4 = delta4(2:end);
 delta3 = (Theta3' * delta4 ) .* transpose(sigmoidGradient(z2(icc,:)));
 delta3 = delta3(2:end);
 delta2 = (Theta2' * delta3 ) .* transpose(sigmoidGradient(z1(icc,:)));
 delta2 = delta2(2:end);
 
 Delta5 = Delta5 + delta6 * a4(icc,:) ; 
 Delta4 = Delta4 + delta5 * a3(icc,:) ;
 Delta3 = Delta3 + delta4 * a2(icc,:) ;
 Delta2 = Delta2 + delta3 * a1(icc,:) ;
 Delta1 = Delta1 + delta2 * Xn(icc,:) ;
end



Theta1_grad=Delta1 / m ;
Theta2_grad=Delta2 / m ;
Theta3_grad=Delta3 / m ;
Theta4_grad=Delta4 / m ;
Theta5_grad=Delta5 / m ;

%with regulariztion
%Theta1_grad=Theta1_grad + lambda/m * ...
 %   [zeros(size(Theta1_grad,1),1) Theta1(:,2:end)];

%Theta2_grad=Theta2_grad + lambda/m * ...
 %   [zeros(size(Theta2_grad,1),1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:) ; Theta4_grad(:); Theta5_grad(:)];


end
