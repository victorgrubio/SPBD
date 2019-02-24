%% Script from slides. LDA applied to two classes
%datos: X1 clase 1 y X2 clase 2
X1=[4,2;2,4;2,3;3,6;4,4]';
X2=[9,10;6,8;9,5;8,7;10,8]';
X=[X1,X2];
figure(1)
stem(X1(1,:),X1(2,:),'r')
hold on
stem(X2(1,:),X2(2,:),'b')
hold off
axis([1 10 0 10])
% class means:
Mu1 = mean(X1,2);
Mu2 = mean(X2,2);
% Covariance matrix of the classes
S1 = cov(X1');
S2 = cov(X2');
% Within-class scatter matrix
Sw = S1 + S2;
% Between-class scatter matrix
SB = (Mu1-Mu2)*(Mu1-Mu2)';
% Computing the LDA projection
invSw = inv(Sw);
invSwSB = invSw*SB;
%getting the projection vector
[V,D] = eig(invSwSB);
% The projection vector
W = V(:,1);
% creating the LDA axis
t=0:20:20;
line_X_M= t*V(1,1);% 1st eigenvector
line_Y_M= t*V(2,1);
line_X_m= t*V(1,2);% 2nd eigenvector
line_Y_m= t*V(2,2);
% Plots
figure(2)
scatter(X1(1,:),X1(2,:),'r');
hold on
scatter(X2(1,:),X2(2,:),'b');
plot(line_X_M,line_Y_M,'k-','LineWidth',3)
plot(line_X_m,line_Y_m,'g-','LineWidth',3)
xlabel('X_1');ylabel('X_2');
axis([-7 10 0 10]); grid on; hold off;
