function [sumSquaredLengths] = SSL(z, u)

yAnon = @(z, u, c) abs(dot(z, u, c)).^2;

totalSum = 0;
for k = 1:92
    totalSum = totalSum + yAnon(z(k, :), u, 2); 
end
sumSquaredLengths = totalSum;
