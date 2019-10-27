function W = randInitializeWeights(L_in, L_out,epsilon_init)

W = zeros(L_out, 1 + L_in);

%
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
