function [answer] = Softmax(x)
tmp = exp(bsxfun(@minus, x, max(x,[],2)));
answer = bsxfun(@rdivide, tmp, sum(tmp, 2));
end