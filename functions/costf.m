function J = costf(theta, xf, yf , lambda,hls,ils,nl)
Theta1f = reshape(theta(1:hls * (ils + 1)), ...
                 hls, (ils + 1));

Theta2f = reshape(theta((1 + (hls * (ils + 1))):end), ...
                 nl, (hls+ 1));
m = size(xf, 1);
Yf=zeros(m,nl);
for iff=1:m
Yf(iff,yf(iff))=1;
end
xf = [ones(m, 1) xf];

z2f=xf * Theta1f' ;
af=sigmoid(z2f);% first layer

af=[ones(size(af,1),1) af];

h_thetaf=sigmoid(af * Theta2f'); % computing htheta (x)
J= -trace( 1/ m * (log(h_thetaf') * Yf + log (1-h_thetaf') * (1-Yf) ));
Theta1_revf=Theta1f(:,2:size(Theta1f,2));
Theta2_revf=Theta2f(:,2:size(Theta2f,2));

J= J + lambda / (2*m) * ...
    (trace(Theta1_revf' *Theta1_revf) + trace(Theta2_revf'*Theta2_revf));