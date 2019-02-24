%% Script addapted from slides. LDA to 3 classes
close all
clc
clear all;
X1=[4,2;2,4;2,3;3,6;4,4]';
X2=[9,10;6,8;9,5;8,7;10,8]';
X3=[14,12;13,12;13,14;13,13;12,14]';
X=[X1,X2,X3];
n1=length(X1);
n2=length(X2);
n3=length(X3);
figure(1)
stem(X1(1,:),X1(2,:),'r')
hold on
stem(X2(1,:),X2(2,:),'b')
stem(X3(1,:),X3(2,:),'m')
hold off
%axis([1 10 0 10])
% class means:
mu1 = mean(X1,2);
mu2 = mean(X2,2);
mu3 = mean(X3,2);
mu = mean(X,2);
% Covariance matrix of the classes
S1 = cov(X1');
S2 = cov(X2');
S3 = cov(X3');
% Within-class scatter matrix
Sw = S1 + S2 + S3;
% Between-class scatter matrix
SB1 = n1*((mu1-mu)*(mu1-mu)');
SB2 = n2*((mu2-mu)*(mu2-mu)');
SB3 = n3*((mu3-mu)*(mu3-mu)');
SB = SB1 + SB2 + SB3;
% Computing the LDA projection
invSw = inv(Sw);
invSwSB = invSw*SB;
%getting the projection vector
[V,D] = eig(invSwSB);
% The projection vector
K=2;
for i=1:K
W(:,i) = V(:,i);
end
% Projection of the classes onto the 2-first vectors LDA
Yc1=W'*X1; 
Yc2=W'*X2;
Yc3=W'*X3;
Yc1_mu=mean(Yc1,2);
Yc2_mu=mean(Yc2,2);
Yc3_mu=mean(Yc3,2);
Yc_mu=(Yc1_mu+Yc2_mu+Yc3_mu)/3;
% Plotting original data
figure(2)
subplot(3,1,1)
scatter(X1(1,:),X1(2,:),'or');
hold on
scatter(X2(1,:),X2(2,:),'^b');
scatter(X3(1,:),X3(2,:),'+m');
plot(mu1(1),mu1(2),'.r','MarkerSize' ,20,'MarkerEdgeColor','r')
plot(mu2(1),mu2(2),'.b','MarkerSize' ,20,'MarkerEdgeColor','b')
plot(mu3(1),mu3(2),'.m','MarkerSize' ,20,'MarkerEdgeColor','m')
plot(mu(1),mu(2),'.k','MarkerSize',20,'MarkerEdgeColor','k')
title('3-classes 2-dimensional data(2nd dimension vs. 1st dimension)')
xlabel('X_1')
ylabel('X_2')
grid on
hold off
%
% Plotting projected data
%figure(3)
mirror=-1;
subplot(3,1,2)
scatter(mirror*Yc1(1,:),Yc1(2,:),'r')
hold on
scatter(mirror*Yc2(1,:),Yc2(2,:),'^b')
scatter(mirror*Yc3(1,:),Yc3(2,:),'+m')
plot(mirror*Yc1_mu(1),Yc1_mu(2),'.r','MarkerSize',20,'MarkerEdgeColor','r')
plot(mirror*Yc2_mu(1),Yc2_mu(2),'.b','MarkerSize',20,'MarkerEdgeColor','b')
plot(mirror*Yc3_mu(1),Yc3_mu(2),'.m','MarkerSize',20,'MarkerEdgeColor','m')
plot(mirror*Yc_mu(1),Yc_mu(2),'.k', 'MarkerSize',20,'MarkerEdgeColor','k')
hold off
title('Projected data onto the 2 largest LDA vectors')
xlabel('Y1')
ylabel('Y2')
% Projection onto the first LDA vector only
%figure(4)
subplot(3,1,3)
Yc1_w1 = W(:,1)'*X1;
Yc2_w1 = W(:,1)'*X2;
Yc3_w1 = W(:,1)'*X3;
hold on
histogram(mirror*Yc1_w1,10,'FaceColor','r')
histogram(mirror*Yc2_w1,10,'FaceColor','b')
histogram(mirror*Yc3_w1,10,'FaceColor','m')
plot(mirror* Yc1_w1,zeros(1,n1),'or')
plot(mirror* Yc2_w1,zeros(1,n2),'^b')
plot(mirror* Yc3_w1,zeros(1,n3),'*m')
title('LDA projected classes onto the biggest eigenvector ONLY')
xlabel('Y')
ylabel('Ocurrences')
hold off