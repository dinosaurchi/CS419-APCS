% Question 1:
a = 1;
b = 100;
x = ([a:b])';

a = 1;
i = -0.1;
b = 0;
y = [a:i:b]';


% Question 2:
n = 25; 
y2 = [x(1:n)];

a = 50;
b = 75;
z = [x(a:b)];

m = length(x) / 2;
w = [x(2*(1:m))];


% Question 3: 
result = ones(3);
result = zeros(8,1);
result = ones(5,2) * 0.37;


% Question 4: 
result = -0.5 * rand(1,25);


% Question 5: 
x = [3,1,2,5,4];
y = fliplr(x);
index = find(y > 2);
z = x(x<4);


% Question 6: 
x = [3,1,2,5,4];
[s, index] = sort(x);
clear x; 
x(index) = s;


% Question 7: 
m = [1,2,3;2,1,5;4,6,4;2,3,2];
n = m([1 2], [1 3]);


% Question 8: 
m = [1,2,3;2,1,5;4,6,4;2,3,2];
n = -1*sortrows(m*-1,2); % Sort with negative sign for descending order, then remove the negative sign


% Question 9: 
x = [1, 2, 3];
y = [0.1, 0.2, 0.3];
result = x * y';

result = x .* y;


% Question 10: 
a = [8,6,4];
n = 4; 
result = (ones(n,1) * a)';


% Question 11: 
a = [0, 2, 1; 3, 1, 0; 4, 6, 4; 2, 0, 2];
result = (a == 0);
b = (ones(size(a,2),1) * max(a'))';
result = a .* (a == b);


% Question 12: 
x = [3, 1, 4];
n = 5; 
y = reshape((ones(n,1) * x), 1, size(x,1)*size(x,2)*n);


% Question 13: 
x = -1:0.1:1;
y = -1:0.1:1;
z = y + x .* exp(-3*abs(y));

[X,Y] = meshgrid(x,y);
Z = Y + X .* exp(-3*abs(Y));
surf(X,Y,Z);
contour(X,Y,Z);


% Question 14:
Z = [1 10 25; 123 233 255; 172 201 54];
result = Z;
[x,i] = sort(Z(:));
[xx,ii] = sort(i);
y = cumsum(diff([0;x])>0);
result(:) = y(ii);









