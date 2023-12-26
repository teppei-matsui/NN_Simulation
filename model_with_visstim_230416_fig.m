
% rmpath(genpath('\\192.168.100.193\public\matlabcodes\matlab_SVNcode_temp'))
% rmpath(genpath('\\192.168.100.193\public\matlabcodes\matlab_SVNcode(do not edit)'))
% rmpath(genpath('D:\public\matlabcodes\matlab_SVNcode_temp'))
% rmpath(genpath('D:\public\matlabcodes\matlab_SVNcode(do not edit)'))
%%
close all;
clear all;

%%
rng(1);

%%
outdir = 'fig_materials';
mkdir(outdir);

file_footer = '.svg';

%%

% clear all;

% Hyper parameters
NCells=180;
Nt=1000;

Anoise1 = 1;
Anoise2 = 0.05;
Anoise3 = 0.05;

threshold_primate_1 = 0.1;
threshold_primate_2 = 0.1;
threshold_primate_3 = 0.1;
threshold_mouse_1 = 0.1;
threshold_mouse_2 = 0.1;
threshold_mouse_3 = 0.1;

% NOris is set equal to NCells
NTrials = 1000; % Number of Trials for Visual Experiment
VisAmp = 9; % Amplitude of Vis Input
VisWidth = 2.5; % width of VonMieses for Vis Input

A_width_primate = 1.8;
A_width_mouse = 0.6;

W = VonMises1_norm([1,1,1], 1:NCells)*0.5;
W2primate = VonMises1_norm([1,A_width_primate,1], 1:NCells);
W2mouse   = VonMises1_norm([1,A_width_mouse,1], 1:NCells);

%%

% spontaneous activity
R1primate=zeros(NCells, Nt);
R2primate=zeros(NCells, Nt);
R3primate=zeros(NCells, Nt);

R1mouse=zeros(NCells, Nt);
R2mouse=zeros(NCells, Nt);
R3mouse=zeros(NCells, Nt);

for t=1:Nt-1
    R1primate(:,t) = Anoise1 * randn(NCells, 1);
    R1primate(:,t) = relu(R1primate(:,t),threshold_primate_1);
    R2primate(:,t) = circ_conv(R1primate(:,t), W) + Anoise2 * randn(NCells,1);
    R2primate(:,t) = relu(R2primate(:,t),threshold_primate_2);
    R3primate(:,t) = circ_conv(R2primate(:,t), W2primate) + Anoise3 * randn(NCells,1);
    R3primate(:,t) = relu(R3primate(:,t),threshold_primate_3);
    
    R1mouse(:,t) = Anoise1 * randn(NCells, 1);
    R1mouse(:,t) = relu(R1mouse(:,t),threshold_mouse_1);
    R2mouse(:,t) = circ_conv(R1mouse(:,t), W) + Anoise2 * randn(NCells,1);
    R2mouse(:,t) = relu(R2mouse(:,t),threshold_mouse_2);
    R3mouse(:,t) = circ_conv(R2mouse(:,t), W2mouse) + Anoise3 * randn(NCells,1);
    R3mouse(:,t) = relu(R3mouse(:,t),threshold_mouse_3);
end

figure

subplot(4,2,1)
imagesc(R2primate(:,101:1000))
title('R2primate')

subplot(4,2,2)
Y=corrcoef(R2primate(:,101:1000)');
image(Y*64)

subplot(4,2,3)
imagesc(R3primate(:,101:1000))
title('R3primate')

subplot(4,2,4)
Y=corrcoef(R3primate(:,101:1000)');
image(Y*64)

subplot(4,2,5)
imagesc(R2mouse(:,101:1000))
title('R2mouse')

subplot(4,2,6)
Y=corrcoef(R2mouse(:,101:1000)');
image(Y*64)

subplot(4,2,7)
imagesc(R3mouse(:,101:1000))
title('R3mouse')

subplot(4,2,8)
Y=corrcoef(R3mouse(:,101:1000)');
image(Y*64)


% Evoked Activity
% インプットがL2に入ると仮定
%

R1primate_vis=zeros(NCells, NCells,NTrials);
R2primate_vis=zeros(NCells, NCells,NTrials);
R3primate_vis=zeros(NCells, NCells,NTrials);

R1mouse_vis=zeros(NCells, NCells,NTrials);
R2mouse_vis=zeros(NCells, NCells,NTrials);
R3mouse_vis=zeros(NCells, NCells,NTrials);


Bnoise1 = Anoise1;
Bnoise2 = Anoise2;
Bnoise3 = Anoise3;

for i = 1:NTrials
    for t=1:NCells %orientation
        
        W = VonMises1_norm([1,1,1], 1:NCells)*0.5;
        VisInput = VisAmp*VonMises1_norm([1,VisWidth,t], 1:NCells)';
        
        R1primate_vis(:,t,i) = Bnoise1 * randn(NCells, 1);
        R1primate_vis(:,t,i) = relu(R1primate_vis(:,t,i),threshold_primate_1);
        R2primate_vis(:,t,i) = circ_conv(R1primate_vis(:,t,i), W) + Bnoise2 * randn(NCells,1) + VisInput;
        R2primate_vis(:,t,i) = relu(R2primate_vis(:,t,i),threshold_primate_2);
        R3primate_vis(:,t,i) = circ_conv(R2primate_vis(:,t,i), W2primate) + Bnoise3 * randn(NCells,1);
        R3primate_vis(:,t,i) = relu(R3primate_vis(:,t,i),threshold_primate_3);
        
        R1mouse_vis(:,t,i) = Bnoise1 * randn(NCells, 1);
        R1mouse_vis(:,t,i) = relu(R1mouse_vis(:,t,i),threshold_mouse_1);
        R2mouse_vis(:,t,i) = circ_conv(R1mouse_vis(:,t,i), W) + Bnoise2 * randn(NCells,1) + VisInput;
        R2mouse_vis(:,t,i) = relu(R2mouse_vis(:,t,i),threshold_mouse_2);
        R3mouse_vis(:,t,i) = circ_conv(R2mouse_vis(:,t,i), W2mouse) + Bnoise3 * randn(NCells,1);
        R3mouse_vis(:,t,i) = relu(R3mouse_vis(:,t,i),threshold_mouse_3);
        
    end
end

R2primate_vis_all = R2primate_vis;
R2mouse_vis_all = R2mouse_vis;
R3primate_vis_all = R3primate_vis;
R3mouse_vis_all = R3mouse_vis;

R2primate_vis_sd = std(R2primate_vis,1,3);
R3primate_vis_sd = std(R3primate_vis,1,3);

R2mouse_vis_sd = std(R2mouse_vis,1,3);
R3mouse_vis_sd = std(R3mouse_vis,1,3);

R2primate_vis = mean(R2primate_vis,3);
R3primate_vis = mean(R3primate_vis,3);

R2mouse_vis = mean(R2mouse_vis,3);
R3mouse_vis = mean(R3mouse_vis,3);

figure()

subplot(4,2,1)
imagesc(R2primate_vis)
title('R2primate')

subplot(4,2,2)
Y=corrcoef(R2primate_vis');
image(Y*64)

subplot(4,2,3)
imagesc(R3primate_vis)
title('R3primate')

subplot(4,2,4)
Y=corrcoef(R3primate_vis');
image(Y*64)

subplot(4,2,5)
imagesc(R2mouse_vis)
title('R2mouse')
subplot(4,2,6)
Y=corrcoef(R2mouse_vis');
image(Y*64)

subplot(4,2,7)
imagesc(R3mouse_vis)
title('R3mouse');
subplot(4,2,8)
Y=corrcoef(R3mouse_vis');
image(Y*64)

%
sponta_primate = R3primate;
vis_primate = R3primate_vis;

sponta_mouse = R3mouse;
vis_mouse = R3mouse_vis;


ori_tuning_width_primate = zeros(2,NCells);
ori_tuning_width_mouse = zeros(2,NCells);
gOSI_primate = zeros(1,NCells);
gOSI_mouse = zeros(1,NCells);
OSI_primate = zeros(1,NCells);
OSI_mouse = zeros(1,NCells);

for c=1:NCells
    ori_tuning_width_primate(1,c) = estimate_ori_tuning_0(R3primate_vis(c,:),1);
    ori_tuning_width_mouse(1,c) = estimate_ori_tuning_0(R3mouse_vis(c,:),1);
    ori_tuning_width_primate(2,c) = estimate_ori_tuning_0(R3primate_vis(c,:),0);
    ori_tuning_width_mouse(2,c) = estimate_ori_tuning_0(R3mouse_vis(c,:),0);
    
    
    % sponta subtraction
    %ori_tuning_width_primate(1,c) = estimate_ori_tuning_0(R3primate_vis(c,:)-mean(R3primate(c,:),2),1);
    %ori_tuning_width_mouse(1,c) = estimate_ori_tuning_0(R3mouse_vis(c,:)-mean(R3mouse(c,:),2),1);
    %ori_tuning_width_primate(2,c) = estimate_ori_tuning_0(R3primate_vis(c,:)-mean(R3primate(c,:),2),0);
    %ori_tuning_width_mouse(2,c) = estimate_ori_tuning_0(R3mouse_vis(c,:)-mean(R3mouse(c,:),2),0);
    
    
    gOSI_primate(c) = vector_average_gOSI(R3primate_vis(c,:));
    gOSI_mouse(c) = vector_average_gOSI(R3mouse_vis(c,:));
    
    OSI_primate(c) = dir_indexKO_OSIs (R3primate_vis(c,:));
    OSI_mouse(c) = dir_indexKO_OSIs (R3mouse_vis(c,:));
end


figure();
set(gcf,'color','w')
set(gcf,'position',[680   558   1200   420])
subplot(1,2,1)
plot(R3mouse_vis(90,:))
set(gca,'xlim',[0 180])
title(['mouse    ',...
    'width(w/): ',num2str(round(median(ori_tuning_width_mouse(1,:)))),...
    ', width(w/o): ',num2str(round(median(ori_tuning_width_mouse(2,:)))),...
    ', OSI: ',num2str(round(median(OSI_mouse)*10)/10),...
    ', gOSI: ',num2str(round(median(gOSI_mouse)*10)/10)]);

subplot(1,2,2)
plot(R3primate_vis(90,:))
set(gca,'xlim',[0 180])
title(['primate    ',...
    'width(w/): ',num2str(round(median(ori_tuning_width_primate(1,:)))),...
    ', width(w/ot): ',num2str(round(median(ori_tuning_width_primate(2,:)))),...
    ', OSI: ',num2str(round(median(OSI_primate)*10)/10),...
    ', gOSI: ',num2str(round(median(gOSI_primate)*10)/10)]);



%
NOris = NCells;
mean_corr_primate = zeros(NOris,1);
mean_corr_mouse = zeros(NOris,1);
all_diff = zeros(NOris,1);

% 方位ごとにresponse patternをmouseとprimateで比較

cumulative_primate = [];
cumulative_mouse = [];

mean_ori_primate = mean(vis_primate,3);
mean_ori_mouse = mean(vis_mouse,3);
for i = 1:Nt-1
    
    temp=zeros(NOris,1);
    for myori = 1:NOris
        r = corrcoef(mean_ori_primate(:,myori),sponta_primate(:,i));
        temp(myori) = r(1,2);
    end
    cumulative_primate = [cumulative_primate, max(abs(temp))];
    
    
    temp=zeros(NOris,1);
    for myori = 1:NOris
        r = corrcoef(mean_ori_mouse(:,myori),sponta_mouse(:,i));
        temp(myori) = r(1,2);
    end
    cumulative_mouse = [cumulative_mouse, max(abs(temp))];
end

% Cumulative plot
xbin = 0.0:0.01:1.0;
[y,x] = hist(cumulative_primate(:),xbin);
[y2,x2] = hist(cumulative_mouse(:),xbin);
figure;hold on;
plot(x,cumsum(y)/sum(y));
plot(x2,cumsum(y2)/sum(y2));
legend({'primate','mouse'});

% Tuning Curves
figure;
subplot(2,2,1);
plot(R2mouse_vis(90,:));
title('R2mouse');
subplot(2,2,2);
plot(R3mouse_vis(90,:));
title('R3mouse');
subplot(2,2,3);
plot(R2primate_vis(90,:));
title('R2primate');
subplot(2,2,4);
plot(R3primate_vis(90,:));
title('R3primate');

% PCA and project variance
[COEFF,score, latent] = pca(sponta_primate');
latent_primate = latent / sum(latent)*100; % percent explained variane
PC_primate = COEFF;

[COEFF,score, latent] = pca(sponta_mouse');
latent_mouse = latent / sum(latent)*100; % percent explained variane
PC_mouse = COEFF;

figure;hold on;
plot(log10(1:NCells),log10(latent_primate));
plot(log10(1:NCells),log10(latent_mouse));

% Project evoked response to first 20 Sponta PCs

nPCs = 20;

projected_primate = zeros(nPCs,NOris);
projected_mouse = zeros(nPCs,NOris);
for i = 1:nPCs
    for j = 1:NOris
        projected_primate(i,j) = sum(mean_ori_primate(:,j).*PC_primate(:,i));
        
        projected_mouse(i,j) = sum(mean_ori_mouse(:,j).*PC_mouse(:,i));
    end
end

total_var_primate = sum(var(mean_ori_primate,1,2));
projected_var_primate = sum(var(projected_primate,1,2));
var_ratio_primate = projected_var_primate/total_var_primate


total_var_mouse = sum(var(mean_ori_mouse,1,2));
projected_var_mouse = sum(var(projected_mouse,1,2));
var_ratio_mouse = projected_var_mouse/total_var_mouse


vis_sponta_ratio_primate = mean(diag(mean(R3primate_vis,3)))/mean(std(R3primate,1,2))
vis_sponta_ratio_mouse = mean(diag(mean(R3mouse_vis,3)))/mean(std(R3mouse,1,2))


r3_sponta_mouse = corrcoef(R3mouse');
r3_sponta_primate = corrcoef(R3primate');
r3_vis_mouse = corrcoef(R3mouse_vis');
r3_vis_primate = corrcoef(R3primate_vis');

R3mouse_vis_all_res = R3mouse_vis_all - repmat(R3mouse_vis,[1,1,NTrials]);
R3primate_vis_all_res = R3primate_vis_all - repmat(R3primate_vis,[1,1,NTrials]);
R3mouse_vis_all_res2 = reshape(R3mouse_vis_all_res,[180,180*NTrials]);
R3primate_vis_all_res2 = reshape(R3primate_vis_all_res,[180,180*NTrials]);

r3_noise_mouse = corrcoef(R3mouse_vis_all_res2');
r3_noise_primate = corrcoef(R3primate_vis_all_res2');

r2_sponta_mouse = corrcoef(R2mouse');
r2_sponta_primate = corrcoef(R2primate');
r2_vis_mouse = corrcoef(R2mouse_vis');
r2_vis_primate = corrcoef(R2primate_vis');

R2mouse_vis_all_res = R2mouse_vis_all - repmat(R2mouse_vis,[1,1,NTrials]);
R2primate_vis_all_res = R2primate_vis_all - repmat(R2primate_vis,[1,1,NTrials]);
R2mouse_vis_all_res2 = reshape(R2mouse_vis_all_res,[180,180*NTrials]);
R2primate_vis_all_res2 = reshape(R2primate_vis_all_res,[180,180*NTrials]);

r2_noise_mouse = corrcoef(R2mouse_vis_all_res2');
r2_noise_primate = corrcoef(R2primate_vis_all_res2');

%%
figure();
imagesc(r3_sponta_mouse);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L3 mouse sponta')
saveas(gcf,fullfile(outdir,['L3 mouse sponta',file_footer]))

figure();
imagesc(r3_sponta_primate);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L3 primate sponta')
saveas(gcf,fullfile(outdir,['L3 primate sponta',file_footer]))

figure();
imagesc(r3_vis_mouse);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-1 1])
title('L3 mouse vis signal')
saveas(gcf,fullfile(outdir,['L3 mouse vis signal',file_footer]))

figure();
imagesc(r3_vis_primate);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-1 1])
title('L3 primate vis signal')
saveas(gcf,fullfile(outdir,['L3 primate vis signal',file_footer]))

figure();
imagesc(r3_noise_mouse);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L3 mouse vis noise')
saveas(gcf,fullfile(outdir,['L3 mouse vis noise',file_footer]))

figure();
imagesc(r3_noise_primate);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L3 primate vis noise')
saveas(gcf,fullfile(outdir,['L3 primate vis noise',file_footer]))

figure();
imagesc(r2_sponta_mouse);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L2 mouse sponta')
saveas(gcf,fullfile(outdir,['L2 mouse sponta',file_footer]))

figure();
imagesc(r2_sponta_primate);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L2 primate sponta')
saveas(gcf,fullfile(outdir,['L2 primate sponta',file_footer]))

figure();
imagesc(r2_vis_mouse);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-1 1])
title('L2 mouse vis signal')
saveas(gcf,fullfile(outdir,['L2 mouse vis signal',file_footer]))

figure();
imagesc(r2_vis_primate);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-1 1])
title('L2 primate vis signal')
saveas(gcf,fullfile(outdir,['L2 primate vis signal',file_footer]))

figure();
imagesc(r2_noise_mouse);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L2 mouse vis noise')
saveas(gcf,fullfile(outdir,['L2 mouse vis noise',file_footer]))

figure();
imagesc(r2_noise_primate);
colormap('jet')
axis image
colorbar
set(gca,'clim',[-0.3 0.3])
title('L2 primate vis noise')
saveas(gcf,fullfile(outdir,['L2 primate vis noise',file_footer]))


figure();
plot(R3mouse_vis(90,:))
set(gca,'xlim',[0 180])
title(['tuning curve mouse']);
saveas(gcf,fullfile(outdir,['tuning curve mouse',file_footer]))

figure();
plot(R3primate_vis(90,:))
set(gca,'xlim',[0 180])
title(['tuning curve primate']);
saveas(gcf,fullfile(outdir,['tuning curve primate',file_footer]))

figure;
plot([1 2],[var_ratio_mouse var_ratio_primate],'o-')
set(gca,'xlim',[0 3],'ylim',[0 1],'xtick',[1,2],'xticklabel',{'primate','mouse'})
title('Projected var / Total var')
saveas(gcf,fullfile(outdir,['var_ratio',file_footer]))

%%
%% Functions
function y = VonMises1_norm(A, xdata)

y = A(1) * exp( A(2) * (cos((xdata-A(3))*pi*2/length(xdata)) -1 ));
y = y./sum(y);

end

% function out = relu(in,threshold)
%     in(find(in-threshold<0))=0;
%     out = in - threshold;
% end

function out = relu(in,threshold)
out = in - threshold;
out = max(out,0);
end

function out = circ_conv(in, ker)
temp=conv([in; in], ker);
out=temp(length(in)+1:length(in)*2);
end

function [tuning_width, best_ori_fit, A1, k1, A, resnorm, residual, exitflag, output] = estimate_ori_tuning_0(ydata,if_baseline_subtraction)

if nargin<2
    if_baseline_subtraction=0;
end

nstim_per_run = length(ydata);

xdata = [0:nstim_per_run-1] * 180 / nstim_per_run;

[ymax, best_ori] = max(ydata);

if if_baseline_subtraction==0
    ymin=0;
elseif if_baseline_subtraction==1
    ymin = min(ydata);
end

A0(1) = ymax - ymin;
A0(2) = 1;
A0(3) = (best_ori -1)*180/nstim_per_run;

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');

fun = @(A, xdata)A(1) * exp( A(2) * (cos((xdata-A(3))*pi/90) -1 ));

[A,resnorm,residual,exitflag,output] = lsqcurvefit(fun, A0, xdata, ydata-ymin, [], [], options);

A1=A(1);
k1=A(2);
best_ori_fit=mod(A(3),180);

if k1>=log(2)/2
    tuning_width = acos(log(0.5)/k1+1)/pi*180/2;
else
    tuning_width= 90;
end
end

function  gOSI = vector_average_gOSI(ydata)

nstim_per_run = length(ydata);

Vx=0;
Vy=0;
sum=0;

for i=1:nstim_per_run
    a = max(ydata(i),0);
    Vx=Vx+a*cos(2*(i-1)*pi/nstim_per_run);
    Vy=Vy+a*sin(2*(i-1)*pi/nstim_per_run);
    sum=sum+a;
end

vector_mag=(Vx^2+Vy^2)^(1/2);
if sum~=0
    gOSI = vector_mag/sum;
else
    gOSI = 0;
end

end

function OSI = dir_indexKO_OSIs (ydata)

nstim_per_run = length(ydata);
[R_best_dir, best_ori] = max(ydata);
null_ori = mod((best_ori + nstim_per_run/2 - 1), nstim_per_run)+1;
R_null_dir = ydata(null_ori);

OSI = (R_best_dir-R_null_dir)/(R_best_dir+R_null_dir);

end

