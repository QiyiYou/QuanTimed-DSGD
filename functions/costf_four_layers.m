function J = costf_four_layers(theta, xf, yf , lambda,il,hl1,hl2,hl3,hl4,num_labels)

             
Theta1f = reshape(theta(1:hl1 * (il + 1)), ...
                 hl1, (il + 1));
i1 = hl1 * (il + 1) ; 
Theta2f = reshape(theta(i1+1:i1+hl2*(hl1+1)), ...
                 hl2, (hl1 + 1));
i2 = i1+hl2*(hl1+1) ;              
Theta3f = reshape(theta(i2+1:i2+hl3 * (hl2 + 1)), ...
                 hl3, (hl2 + 1));
i3 = i2+hl3 * (hl2 + 1) ;             
Theta4f = reshape(theta(i3+1:i3+hl4 * (hl3 + 1)), ...
                 hl4, (hl3 + 1));
i4 = i3+hl4 * (hl3 + 1) ;
Theta5f = reshape(theta(i4+1:i4+num_labels * (hl4 + 1)), ...
                 num_labels, (hl4 + 1));               
             
             
m = size(xf, 1);
Yf = zeros(m,num_labels);
for iff=1:m
Yf(iff,yf(iff))=1;
end
xf = [ones(m, 1) xf];

z1f = xf * Theta1f' ;
a1f = sigmoid(z1f);% first layer
a1f = [ones(size(a1f,1),1) a1f];

z2f = a1f * Theta2f' ;
a2f = sigmoid(z2f);% first layer
a2f = [ones(size(a2f,1),1) a2f];

z3f = a2f * Theta3f' ;
a3f = sigmoid(z3f);% first layer
a3f = [ones(size(a3f,1),1) a3f];

z4f = a3f * Theta4f' ;
a4f = sigmoid(z4f);% first layer
a4f = [ones(size(a4f,1),1) a4f];

h_thetaf=sigmoid(a4f * Theta5f'); % computing htheta (x)
J = -trace( 1/ m * (log(h_thetaf') * Yf + log (1-h_thetaf') * (1-Yf) ));
%Theta1_revf=Theta1f(:,2:size(Theta1f,2));
%Theta2_revf=Theta2f(:,2:size(Theta2f,2));

%J= J + lambda / (2*m) * ...
 %   (trace(Theta1_revf' *Theta1_revf) + trace(Theta2_revf'*Theta2_revf));