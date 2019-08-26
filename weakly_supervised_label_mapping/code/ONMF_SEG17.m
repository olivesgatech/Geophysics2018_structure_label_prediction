function [W, H] = ONMF_SEG17(X,W_,H_,B,sW,N_l,save_memory)
% This code takes in: 
% 1- the data matrix X
% 2- the initialized feature matrix W_
% 3- the initialized coefficient matrix H_
% 4- matrix B used for the orthogonality constraint
% 5- the sparsity level of the features: sW
% 6- the number of labels
% 7- save_memory is an flag. 0: saves all intermediary results. 1(default):
% saves only the latest result (to save memory)
% -----------------------------------------------------------------------
% The output: 
% 1- The final feature matrix W
% 2- The final coefficients matrix H
% -----------------------------------------------------------------------

% --------------------------------------------------------------------- %
%                              Main Parameters                          %
% --------------------------------------------------------------------- %
W_ = sparsify_columns(W_, sW);

maxIter = 400;      % Default: 300
eps     =  1e-5;    % to avoid division by zero 
lambda1 = 0.1;      % Default: 0.1 -- constraint on F-norm of W
lambda2 = 0.5;      % Default: 0.5 -- constraint on F-norm of H
gamma1  = 5;        % Default: 5 (as long as you're normalizing HW) -- Orthogonality constraint on H 
normalizeHW = 1;    % Default: 1 - Normalizes the values of H and W every iteration

% Note: Code converges faster *without* normalization. Takes longer to
% converge when you normalize, but usually leads to better results. 

% set variables and matrices:
[N_p, N_s] = size(X);   % N_p: number of pixels, N_s: number of images (samples)
[~, K] = size(W_);      % K: total # of features/atoms
k = K/N_l;              % k: number of features/atoms per class

% Since the sparsity constraint is applied only once, we should disable the
% sparsity constraint: 
sW = 0;

if ~save_memory
    W = zeros(N_p,K,maxIter);
    H = zeros(K,N_s,maxIter);
    W(:,:,1) = W_;
    H(:,:,1) = H_;
else
    W = W_;
    H = H_;
end

% --------------------------------------------------------------------- %
%                              ONMF Main Code                           %
% --------------------------------------------------------------------- %

gcf = figure(1);
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.

WUpdateNorm = zeros(1,maxIter);
HUpdateNorm = zeros(1,maxIter);

WObjective = zeros(1,maxIter);
HObjective = zeros(1,maxIter);

WNorm = zeros(1,maxIter);
HNorm = zeros(1,maxIter);

stepsizeW = 1;

for iter = 2:maxIter
  
    % set old values:
    if ~save_memory
        W_ = squeeze(W(:,:,iter-1));
        H_ = squeeze(H(:,:,iter-1));
    else
        W_ = W;
        H_ = H;
    end
    
    % ------------------------ STEP 1: Update W -------------------------
    
    if sW == 0 % if no sparsity constraint
        % standard multiplicative update rule
        W_t = W_.*((X*H_'+ eps)./(W_*H_*H_' + lambda1*W_ + eps));
    else % impose sparsity constraint
        % Gradient for W
        dW = (W_*H_-X)*H_';
        begobj = norm(X-W_*H_,'fro')^2; 
        idx = 0;
        % Make sure we decrease the objective!
        while 1
            idx = idx+1;
            % Take step in direction of negative gradient, and project
            W_new = W_ - stepsizeW*dW;
            L2norms = sqrt(sum(W_new.^2));
            % impose sparisty const. on  W:
            for i=1:K
                % compute sparse col and update
                W_new(:,i) = projfunc(W_new(:,i),L1L2ratio*L2norms(i),(L2norms(i)^2),1,sW);
            end
            
            % Calculate new objective
            newobj = norm(X-W_new*H_,'fro')^2; 
            
            % If the objective decreased, we can continue:
            disp(['iter:', num2str(idx), 'newobj-begobj = ', num2str(newobj-begobj)]);
            if newobj<=begobj,
                break;
            end
            
            % else decrease stepsize and try again:
            stepsizeW = stepsizeW/2;
            if stepsizeW<1e-200, % this might be too strict :/
                fprintf('Algorithm converged.\n');
                return;
            end
            
        end
        
        % Slightly increase the stepsize
        stepsizeW = stepsizeW*1.2;
        W_t = W_new;
        
    end
        
    WUpdateNorm(iter) = norm(W_t-W_,2);
    WObjective(iter) = norm(X-W_t*H_,'fro')^2 + (1-ceil(sW))*lambda1*norm(W_t,'fro')^2;
    WNorm(iter) = norm(W_t,'fro');
    
    % ------------------------ STEP 2: Update H --------------------------
    
    H_t = H_.*((W_t'*X + gamma1*(B+B')*H_ + eps)./(W_t'*W_t*H_+ gamma1*(H_*H_'*H_) + lambda2*H_ + eps));
    
    if normalizeHW == 1
        % Renormalize so columns of W and rows of H have constant energy
        norms_row_H = sqrt(sum(transpose(H_t.^2)))';
        H_t = H_t ./ (norms_row_H*ones(1,size(H,2)));
        norms_col_W = sqrt(sum(W_t.^2));
        W_t = W_t ./ (ones(size(W,1),1)*norms_col_W);
    end
        
    % save result:
    if ~save_memory
        H(:,:,iter) = H_t;
        W(:,:,iter) = W_t;
    else
        H = H_t;
        W = W_t;
    end

    HUpdateNorm(iter) = norm(H_t-H_,2);
    HObjective(iter) = norm(X - W_t*H_t,'fro')^2 + lambda2*norm(H_t,'fro')^2 + gamma1*norm(H_t*H_t' - B,'fro')^2 ;
    HNorm(iter) = norm(H_t,'fro');
    
    drawnow;
    
    subplot(421)
    semilogy(1:numel(WUpdateNorm), WUpdateNorm, 'b-','LineWidth',2);
    ylabel('WUpdateNorm','fontsize',18);
    xlabel('iteration','fontsize',18);
    grid on;
    subplot(423)
    semilogy(1:numel(WNorm), WNorm, 'r-','LineWidth',2);
    ylabel('WNorm','fontsize',18);
    xlabel('iteration','fontsize',18);
    grid on;
    subplot(425)
    semilogy(1:numel(HUpdateNorm) , HUpdateNorm, 'g-','LineWidth',2);
    ylabel('HUpdateNorm','fontsize',18);
    xlabel('iteration','fontsize',18);
    grid on;
    subplot(427)
    semilogy(1:numel(HNorm) , HNorm, 'm-','LineWidth',2);
    ylabel('HNorm','fontsize',18);
    xlabel('iteration','fontsize',18);
    grid on;
    
    subplot(422)
    semilogy(1:numel(WObjective), WObjective, 'b-','LineWidth',2);
    ylabel('W Objective','fontsize',18);
    xlabel('iteration','fontsize',18);
    grid on;
    subplot(424)
    imagesc(W_t);
    ylabel('W_t','fontsize',18);
    subplot(426)
    semilogy(1:numel(HObjective) , HObjective, 'g-','LineWidth',2);
    ylabel('H Objective','fontsize',18);
    xlabel('iteration','fontsize',18);
    grid on;
    subplot(428)
    imagesc(H_t);
    ylabel('H_t','fontsize',18);
    xlabel(['Orthogonality term norm: ', num2str(norm(norm(H_t*H_t' - B,'fro')^2))]);
end
end


