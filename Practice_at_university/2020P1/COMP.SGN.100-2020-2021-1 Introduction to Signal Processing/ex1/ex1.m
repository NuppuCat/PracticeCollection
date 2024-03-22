%task1
disp('task1');
x = [1 3 0 -1 5];
disp(x);
whos;
a = [5,6,7]
x = [130- 15];
a = [1 2 3];
b = [4 5];
c = [a -b];
a = [1 3 7];
a = [a 0 -1];
X = [];
%t2 
disp('task2');
disp('2.1');
2/2*3
2/3^2
(2/3)^2
2+3*4-4
2^2*3/3+3
2^(2*3)/(4+3)
2*3+4
2^3^2
-4^2
disp('2.2');
sqrt(2)
(3+4)/(5+6)
(5+3)/(5*3)
2^(3^2)
(2*pi)^2
2*pi^2
1/(sqrt(2*pi))
1/(2*sqrt(pi))
(2.3*4.5)^(1/3)
(1-2/(3+2))/(1+2/(3-2))
1000*(1 + 0.15/12)^60
(0.0000123+5.678*10^(-3))*0.4567*10^(-4)

%task3
disp('task3');
tempSum = 0;
for i=1:1000
  tempSum = tempSum + i^(2); 
end
tempSum

tempSum2 = 0;
i = 1;
j = 0;
while i<1004
    
   tempSum2 = tempSum2 + (-1)^j/i;
   j = j + 1;
   i = i+2;
end
tempSum2

tempSum3 = 0;
i = 1;
j = 0;
while j<501
    
   tempSum3 = tempSum3 + 1/(i^2*(i+2)^2);
   j = j + 1;
   i = i+2;
end
tempSum3

x = 3>2
x = 2>3
x = -4 <= -3
x = 1<1
x = 2~= 2
x = 3== 3
x = 0< 0.5<1

%task 4
disp('task2');
9,87
.0
25.82
-356231
%未定义函数或变量 Undefined function or variable'e2 3.57*e2
disp('Undefined function or variable e2 3.57*e2');
%表达式无效 Invalid expression  3.57e2.1
disp('Invalid expression  3.57e2.1');
3.57e+2
3,57e-2

a2 = 1
%表达式无效 Invalid expression a.2 = 1
disp('Invalid expression a.2 = 1');
%表达式无效 Invalid expression 2a = 1
disp('Invalid expression 2a = 1');
%表达式无效 Invalid expression 'a'one = 1
disp("Invalid expression 'a'one = 1");
aone = 1;
%表达式无效 Invalid expression _x_1 = 1
disp('Invalid expression _x_1 = 1');
miXedUp = 1
%未定义函数或变量 Undefined function or variable 'pay'。 pay day = 1
disp(' Undefined function or variable pay day = 1');
inf = 1
Pay_Day = 1
%'=' 运算符的使用不正确 using '=' incorrectly min*2 =1 
disp(" using '=' incorrectly min*2 =1");
what =1

% p+w/u

% p+w/(u+v)

% (p+w/(u+v))/(p+w/(u-v))
% x^(1/2)
% y^(y+z)
% x^(y^z)
% (x^y)^z
%x-x^3/factorial(3)+x^5/factorial(5)

% i = i+1;
% i = i^3+j
% g = max(e,f);

% if d>0
%    x = -b; 
% end

%x =(a+b)/(c*d)

%2.5
%using '=' incorrectly n+1 =n
%unvalid variblename,cannot find function or varible Fahrenheit temp =
%9*C/5+32
%using '=' incorrectly 2 = x;

%2.6
a = 2;
b = -10;
c = 12;
x = (-b+(b^2-4*a*c)^(0.5))/(2*a)

%2.7
litters = ConvertWeight(2,4)
%Task5
disp('task5');
%2.20
v = [3 1 5];
i = 1;
for j = v
    i = i + 1;
    if i == 3
        i = i + 2;
        m = i + j;
    end
end
i
j
m
%loop 0  1   2     3
% j      3   1     5
% i   1  2   3 5   6
% m      5   6    

%Task6
disp('task6');
x =10;
t = [1790:10:2250];
Pt = 197273000./(1+exp(-0.03134*(t-1913.25)));
figure(1);
plot(t,Pt);
xlabel('year'),ylabel('population')
for i = t
    [m,n] = find(t==i);
    disp(['The population in',num2str(i),' is ', num2str(Pt(n))]);
end

%task 7 
disp('task7');
celcius = fahrenheit_to_celcius(98.6)

%task8
disp('task8');
x = 0:0.01:2*pi;
y = cos(x);
figure(2);
plot(x,y);

%task9
disp('task9');
x =10;
t = [1790:10:2000];
Pt = 197273000./(1+exp(-0.03134*(t-1913.25)));
figure(3);
plot(t,Pt);
hold on;
xlabel('year'),ylabel('population')
actualData = 1000*[3929 5308 7240 9638 12866 17069 23192 31443 38558 50156 62948 75995 91972 105711 122775 131669 150697];

for i = 1:length(actualData)
    %actualData(i)
    plot(t(i),actualData(i),'o');
   
end

%task10
disp('task10');
figure(4);
drawSpiral(0.1,1.4);