function [best, bestFit, convergeVal, stdPerIter, averageFit, firstDVal] = ...
    AOW_D(popSize,dim, pLv, sr, maxIter,ub,lb,fobj)

if isscalar(ub)
    ub = ones(1,dim) * ub;
end

if isscalar(lb)
    lb = ones(1,dim) * lb;
end

isNewFitnestBetter = @(newFit, oldFit) newFit < oldFit;

% Initialization
pop = (ub - lb) .* rand(popSize, dim) + lb;
fit = zeros(1, popSize);
for popIdx = 1:popSize
    fit(popIdx) = fobj(pop(popIdx,:));
end

% Best agent.
[bestFit, bestIdx] = min(fit);
best = pop(bestIdx,:);

[worstFit, ~] = max(fit);

iter = 0;

while iter < maxIter
    E = 1 - iter/maxIter;
    
    % Update population
    newPop = zeros(popSize,dim);
    for popIdx = 1:popSize
        if E > pLv
            if rand > sr
                r = randperm(popSize, 2);
                newPop(popIdx,:) = pop(popIdx,:) + rand * (pop(r(1),:) - pop(r(2),:));
            else
                lst = setdiff(1:popSize, [popIdx, bestIdx]);
                if popIdx == bestIdx
                    spyAgentIdx = lst(randperm(popSize-1, 2));
                else
                    spyAgentIdx = lst(randperm(popSize-2, 2));
                end
                
                spyAgent = pop(spyAgentIdx(1), :);
                pushAgent = pop(spyAgentIdx(2), :);
                
                r = rand(3,dim);
                
                trLB = 1 - ((fit(popIdx) - bestFit) / (worstFit - bestFit + eps));
                tr = (1-trLB) .*  rand(2, dim) + trLB;
                
                newPop(popIdx,:) = pop(popIdx,:) + E * r(1,:) .* (spyAgent - pop(popIdx,:)) ...
                    + r(2,:) .* (tr(1,:) .* best - spyAgent) ...
                    - r(3,:) .*(tr(2,:) .* pushAgent - spyAgent);
            end
        else
            newPop(popIdx,:) = pop(popIdx,:) + E * Levy(1,dim);
        end
    end
    
    % Boundary limit
    for popIdx = 1:popSize
        Flag4ub = newPop(popIdx,:) > ub;
        Flag4lb = newPop(popIdx,:) < lb;
        newPop(popIdx,:) = (newPop(popIdx,:) .* ...
            (~ (Flag4ub + Flag4lb))) + ub .* Flag4ub+lb .* Flag4lb;
    end
    
    % Fitness checking
    newFit = zeros(1, popSize);
    for popIdx = 1:popSize
        newFit(popIdx) = fobj(newPop(popIdx,:));
    end
    
    FE = FE + popSize;
    
    for popIdx = 1:popSize
        if isNewFitnestBetter(newFit(popIdx), fit(popIdx))
            pop(popIdx,:) = newPop(popIdx,:);
            fit(popIdx) = newFit(popIdx);
        end
    end
    
    % Find new best and worst
    [bestFit, bestIdx] = min(fit);
    best = pop(bestIdx,:);
    
    [worstFit, ~] = max(fit);
    
    convergeVal(iter+1) = bestFit;
    stdPerIter(iter+1) = std(fit);
    averageFit(iter+1) = mean(fit);
    firstDVal(iter+1) = mean(pop(:, 1));
    
    iter = iter + 1;
end
end

% Levy flights by Mantegna's algorithm
function levyNum=Levy(m,n)
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
levyNum = zeros(m,n);
for j=1:m
    u=randn(1,n)*sigma;
    v=randn(1,n);
    levyNum(j,:)=u./abs(v).^(1/beta);
end
end
