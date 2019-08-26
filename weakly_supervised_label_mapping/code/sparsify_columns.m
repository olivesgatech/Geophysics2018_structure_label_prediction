function sparse_matrix = sparsify_columns(matrix, sparsity)
% Written by Yazeed Alaudah -- Dec 2016

N_p = size(matrix,1);
% desired L1 to L2 ratio to acheive sparsity level:
L1L2ratio = sqrt(N_p) - sqrt(N_p-1)*sparsity;
L2W = sqrt(sum(matrix.^2)); % L2 norms of columns of matrix
% impose sparisty const. on initial W:
for i =1:size(matrix,2)
    col = matrix(:,i);
    % compute sparse col
    scol = projfunc(col,L1L2ratio*L2W(i),L2W(i)^2,1,sparsity);
    % update:
    matrix(:,i) = scol;
end
sparse_matrix = matrix;