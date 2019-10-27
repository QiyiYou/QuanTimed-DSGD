function grad = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda,td,vmin,vmax)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
i= floor((rand (1) * (vmax-vmin) + vmin)*td) ;
if (i==0)
    grad= zeros(size(Theta1_grad,1)*size(Theta1_grad,2)+size(Theta2_grad,1)*size(Theta2_grad,2),1);
else
r= randi(size(X,1),i,1);
%Xn=zeros (length(r),size(X,2));
%yn=zeros (length(r),1);
Xn= X(r,:);
yn= y(r,:);
m = size(Xn, 1);


X_copy=Xn;
Xn = [ones(m, 1) Xn];

z2 = Xn * Theta1' ;
a = sigmoid(z2);% first layer

a = [ones(size(a,1),1) a];

h_theta=sigmoid(a * Theta2'); % computing htheta (x)

Y=zeros(m,num_labels);

% generating Matrix Y from Vector y:
for ic=1:m
Y(ic,yn(ic))=1;
end
% computing cost function from the computed matrix Y:
%J= -trace( 1/ m * (log(h_theta') * Y + log (1-h_theta') * (1-Y) )); 


% with regularization
% first removing the first column
%Theta1_rev=Theta1(:,2:size(Theta1,2));
%Theta2_rev=Theta2(:,2:size(Theta2,2));

%J= J + lambda / (2*m) * ...
%    (trace(Theta1_rev' *Theta1_rev) + trace(Theta2_rev'*Theta2_rev));

z2=[ones(size(z2,1),1) z2];
Delta2=zeros(size(Theta2,1),size(Theta1,1)+1);
Delta1=zeros(size(Theta1,1),size(X_copy,2)+1);
for icc=1:m
 delta3= transpose(h_theta(icc,:) - Y(icc,:));
 delta2= (Theta2' * delta3 ) .* transpose(sigmoidGradient(z2(icc,:)));
 delta2=delta2(2:end);
 Delta2=Delta2 + delta3 * (a(icc,:));
 Delta1=Delta1 + delta2 * (Xn(icc,:));
end



Theta1_grad=Delta1 / m ;
Theta2_grad=Delta2 / m ;

%with regulariztion
Theta1_grad=Theta1_grad + lambda/m * ...
    [zeros(size(Theta1_grad,1),1) Theta1(:,2:end)];

Theta2_grad=Theta2_grad + lambda/m * ...
    [zeros(size(Theta2_grad,1),1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
