%% Applications of Tucker decomposition
% We will use MLSVD-Tucker3 modeling to investigate a data set from 
% spectrofluorometric analysis of thick juice, an intermediary sugar product.
% The dimensions of the array is (28,20,311). The first mode is fraction 
% number (or elution time), the second mode is the excitation wavelength 
% (250 nm - 440 nm) and the third mode is the emission wavelength 
% (250 nm - 560 nm). Start by loading the data set.
% Load dataset1_st and inspect fluorescence landscapes of the 28 fractions. 
% Look for features/patterns in the modes that could be exploited. 
%
close all
clear all
load('dataset1_st');
%%
fprintf('\n 1 Pak ### Inspect raw data - press a key..\n');pause
figure(1);set(gcf,'Position',[-1 31 804 534]);
for i=1:DimX(1),
   m=reshape(X(i,:),DimX(2),DimX(3));
   mesh(EmAx,ExAx,m);
   title(['Raw data. Time ' int2str(i)]);
   xlabel('Excitation [nm]')
   ylabel('Emission [nm]')
   axis([EmAx(1) EmAx(DimX(3)) ExAx(1) ExAx(DimX(2)) 0 1000]);
   grid on
   drawnow
   pause(.5)
end;
%%
fprintf('\n 2 Pak ### Inspect calibration data - press a key..\n');pause
%Removed wrong/disturbing observations
figure(1);set(gcf,'Position',[-1 31 804 534]);
for i=1:DimX(1),
   mnr=Xc(i,:,:); % Corrected Tensor X (removed disturbing observations)
   mesh(EmAx,ExAx,squeeze(mnr));
   title(['Corrected data. Time ' int2str(i)]);
   xlabel('Excitation [nm]')
   ylabel('Emission [nm]')
   axis([EmAx(1) EmAx(DimX(3)) ExAx(1) ExAx(DimX(2)) 0 1000]);
   grid on
   drawnow
   pause(.5)
end;
%%
fprintf('\n 3 Pak ### Calculate all possible 1-4 models and list sorted - press a key\n');pause
%
Wmax=[4,4,4];
p=0;
options=struct;
options.Normalize=true;
for i=1:Wmax(1),
    for j=1:Wmax(2),
        for k=1:Wmax(3), 
            W=[i j k];
            if prod(W)/max(W)>=max(W),
                W; p=p+1;
                Z(p,1:3)=W;
                %[Factors,G,SSEAux]=lmlra(X,W,options);
                [Factors,G]=mlsvd(X,W,options);
                SSE(p)=100*(1-(frob(X-lmlragen(Factors,G))/frob(X))^2);
            end;
        end;
    end;
end;

[SSE_sorted j_sorted]=sort(SSE);
plot(SSE(j_sorted));grid on;
for i=1:length(SSE(:));
    text(i,SSE(j_sorted(i)),['(',num2str(Z(j_sorted(i),:)),')']);
end;
title('SSEx as function of Tucker3 model dimensionality');
xlabel('Tucker3 model dimensionality (sorted)');
ylabel('Explained variation of X');
%
for i=1:length(SSE),
   fprintf(' %2i  [%i %i %i]  %f \n',i,Z(j_sorted(i),:),SSE(i));
end;
%%
W=[1 3 3];
fprintf('\n 4 Pak ### Calculate a [1 3 3] model, list core and facplot - press a key\n');pause

format short
format compact

[Factors,G]=mlsvd(X,W);
int2str(G)
figure(1);set(gcf,'Position',[-1 31 804 534]);
A = Factors{1}; B = Factors{2}; C = Factors{3};
figure(1);set(gcf,'Position',[-1 31 804 534]);
subplot(2,2,1),plot(A),title('Factors in first mode') %Plot the first component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,2),plot(B),title('Factors in second mode') %Plot the second component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,3),plot(C),title('Factors in third mode') %Plot the third component matrix�� 
axis tight, grid on,legend('1','2','3')
%%
W=[3 3 3];
fprintf('\n 5 Pak ### Continue with a [3 3 3] model, list core and facplot - press a key\n');pause

[Factors,G]=mlsvd(X,W);
disp('[3 3 3] Before rotation')
int2str(G)
A = Factors{1}; B = Factors{2}; C = Factors{3}; % Matrices Factors
figure(1);set(gcf,'Position',[-1 31 804 534]);
subplot(2,2,1),plot(A),title('Factors in first mode') %Plot the first component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,2),plot(B),title('Factors in second mode') %Plot the second component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,3),plot(C),title('Factors in third mode') %Plot the third component matrix�� 
axis tight, grid on,legend('1','2','3')
%%
fprintf('\n 6 Pak ### Continue with a [3 3 3] model with VOS rotation and facplot  - press a key\n');pause

disp('[3 3 3] After rotation to optimal variance of squares')

Av=A*Ov1;Bv=B*Ov2;Cv=C*Ov3; % rotation of factors A, B, C to obtain Av, Bv, Cv
Gv = -ones(3,3,3); % Counter-rotation of tensor G to obtain tensor Gv
Gv(:,:,1) = G(:,:,1)*Ov1;
Gv(:,:,2) = G(:,:,2)*Ov2;
Gv(:,:,3) = G(:,:,3)*Ov3;
SSres = frob((X-lmlragen({Av,Bv,Cv},Gv)))^2;
figure(2);set(gcf,'Position',[-1 31 804 534]);
subplot(2,2,1),plot(Av),title('Factors in first mode') %Plot the first component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,2),plot(Bv),title('Factors in second mode') %Plot the second component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,3),plot(Cv),title('Factors in third mode') %Plot the third component matrix�� 
axis tight, grid on,legend('1','2','3')
%%
fprintf('\n 7 Pak ### The [3 3 3] model with DIA rotation and facplot - press a key\n');pause

disp('[3 3 3] After rotation to optimal diagonality')

Ad=A*Od1;Bd=B*Od2;Cd=C*Od3;% rotation of factors A, B, C to obtain Ad, Bd, Cd
Gd = -ones(3,3,3); % Counter-rotation of tensor G to obtain tensor Gv
Gd(:,:,1) = G(:,:,1)*Od1;
Gd(:,:,2) = G(:,:,2)*Od2;
Gd(:,:,3) = G(:,:,3)*Od3;
SSres = frob((X-lmlragen({Ad,Bd,Cd},Gd)))^2;
figure(3);set(gcf,'Position',[-1 31 804 534]);
subplot(2,2,1),plot(Ad),title('Factors in first mode') %Plot the first component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,2),plot(Bd),title('Factors in second mode') %Plot the second component matrix�� 
axis tight, grid on, legend('1','2','3')
subplot(2,2,3),plot(Cd),title('Factors in third mode') %Plot the third component matrix�� 
axis tight, grid on,legend('1','2','3')
%%
fprintf('\n 8 Pak ### Compare cores  - press a key\n');pause

fprintf('\nNot rotated - Fig 1\n');disp(int2str(G));
fprintf('\nVOS rotated - Fig 2\n');disp(int2str(Gv));
fprintf('\nDIA rotated - Fig 3\n');disp(int2str(Gd));

format