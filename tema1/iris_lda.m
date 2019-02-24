%Script from slides to apply LDA to IRIS dataset
%data: Read the iris data set
close all
clc
clear all
%
filename='iris_data.csv';
fid=fopen(filename,'r');
X=textscan(fid,'%f %f %f %f %*[^\n]','Delimiter',',');
fclose(fid);
%celldisp(X)
X1=X{1};X2=X{2};X3=X{3};X4=X{4};
X=[X1,X2,X3,X4]';
siz=size(X);
% Splitting data in classes
Xc1=X(:,1:50);Xc2=X(:,51:100);
Xc3=X(:,100:150);
n1=length(Xc1);
n2=length(Xc2);
n3=length(Xc3);
%N=n1+n2+n3;
M=siz(1);
N=siz(2);
C=3;
% class means:
Mu1 = mean(Xc1,2);
Mu2 = mean(Xc2,2);
Mu3 = mean(Xc3,2);
Mu = mean(X,2);
% Covariance matrix of the classes
S1 = cov(Xc1');
S2 = cov(Xc2');
S3 = cov(Xc3');
% Within-class scatter matrix
Sw = S1 + S2 + S3;
% Between-class scatter matrix
SB1 =n1*( (Mu1-Mu)*(Mu1-Mu)');
SB2 =n2*( (Mu2-Mu)*(Mu2-Mu)');
SB3 =n3*( (Mu3-Mu)*(Mu3-Mu)');
SB=SB1+SB2+SB3;
% Computing the LDA projection
invSw = inv(Sw);
invSwSB = invSw*SB;
%getting the projection vector
[V,D] = eig(invSwSB);
disp('The diagonal EIV matrix is: ')
disp(num2str(D))
% The projection vector
K=2;
for i=1:K
W(:,i) = V(:,i);
end
Yc1=W'*Xc1; % Projection of the classes onto the 2-first vectors LDA
Yc2=W'*Xc2;
Yc3=W'*Xc3;
Yc1_Mu=mean(Yc1,2);
Yc2_Mu=mean(Yc2,2);
Yc3_Mu=mean(Yc3,2);
Yc_Mu=(Yc1_Mu+Yc2_Mu+Yc3_Mu)/3;
% Plotting original data
figure(2)
subplot(3,1,1)
scatter(Xc1(1,:),Xc1(2,:),'or');
hold on
scatter(Xc2(1,:),Xc2(2,:),'^b');
scatter(Xc3(1,:),Xc3(2,:),'+m');
plot(Mu1(1),Mu1(2),'.r','MarkerSize' ,20,'MarkerEdgeColor','r')
plot(Mu2(1),Mu2(2),'.b','MarkerSize' ,20,'MarkerEdgeColor','b')
plot(Mu3(1),Mu3(2),'.m','MarkerSize' ,20,'MarkerEdgeColor','m')
plot(Mu(1),Mu(2),'.k','MarkerSize',20,'MarkerEdgeColor','k')
title('3-classes 4-dimensional data(2nd dimension vs. 1st dimension)')
xlabel('X_1')
ylabel('X_2')
axis([4 8 2 5])
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
plot(mirror*Yc1_Mu(1),Yc1_Mu(2),'.r','MarkerSize',20,'MarkerEdgeColor','r')
plot(mirror*Yc2_Mu(1),Yc2_Mu(2),'.b','MarkerSize',20,'MarkerEdgeColor','b')
plot(mirror*Yc3_Mu(1),Yc3_Mu(2),'.m','MarkerSize',20,'MarkerEdgeColor','m')
plot(mirror*Yc_Mu(1),Yc_Mu(2),'.k', 'MarkerSize',20,'MarkerEdgeColor','k')
hold off
title('Projected data onto the 2 largest LDA vectors')
xlabel('Y1')
ylabel('Y2')
axis([-3 3 1 3])
% Projection onto the first LDA vector only
%figure(4)
subplot(3,1,3)
Yc1_w1 = W(:,1)'*Xc1;
Yc2_w1 = W(:,1)'*Xc2;
Yc3_w1 = W(:,1)'*Xc3;
hold on
histogram(mirror*Yc1_w1,10,'FaceColor','r')
histogram(mirror* Yc2_w1,10,'FaceColor','b')
histogram(mirror* Yc3_w1,10,'FaceColor','m')
plot(mirror* Yc1_w1,zeros(1,n1),'or')
plot(mirror* Yc2_w1,zeros(1,n2),'^b')
plot(mirror* Yc3_w1,zeros(1,n3),'*m')
title('LDA projected classes onto the biggest eigenvector ONLY')
xlabel('Y')
ylabel('Ocurrences')
hold off