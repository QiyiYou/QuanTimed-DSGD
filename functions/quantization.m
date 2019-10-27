function output = quantization(input,s)

    l=10^(s);
    %output = zeros(size(input));
    quan = round(abs(input),s);   
    randomnoise = rand (size(input)) ;
    r = randomnoise < (abs(abs(input)-quan)*l) ;
    output = sign(abs(input)-quan) .* r/l +quan ;
    output = sign(input) .*output ;
end