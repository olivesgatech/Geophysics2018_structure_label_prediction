function output = normalizeColumns(input)
% normalizes the input matrix to have the L1 norm of each column = 1
output = zeros(size(input));
for i = 1:size(input,2)
    output(:,i) = input(:,i)./sum(abs(input(:,i)));
end

end