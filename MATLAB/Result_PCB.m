clear all
close all
clc

%%
clims = [];
image_save = 0;

img_noisy = rot90(double(importdata('./PCB/noisy4.mat')).*255,2);
img_domain = rot90(double(importdata('./PCB/dynamicmask4.mat')).*255,2);
img_jitter = rot90(double(importdata('./PCB/jitter4.mat')).*255,2);
img_BM3D = rot90(double(importdata('./PCB/BM3D4.mat')).*255,2);
img_unet = rot90(double(importdata('./PCB/unet4.mat')).*255,2);

mask_red = [295 1455 199 199];
mask_green = [556 611 599 599];
mask_yellow = [831 791 100 300];

img_domain_red = img_domain(mask_red(2):mask_red(2)+mask_red(4),mask_red(1):mask_red(1)+mask_red(3));
img_jitter_red = img_jitter(mask_red(2):mask_red(2)+mask_red(4),mask_red(1):mask_red(1)+mask_red(3));
img_BM3D_red = img_BM3D(mask_red(2):mask_red(2)+mask_red(4),mask_red(1):mask_red(1)+mask_red(3));
img_unet_red = img_unet(mask_red(2):mask_red(2)+mask_red(4),mask_red(1):mask_red(1)+mask_red(3));
img_noisy_red = img_noisy(mask_red(2):mask_red(2)+mask_red(4),mask_red(1):mask_red(1)+mask_red(3));

img_domain_green = img_domain(mask_green(2):mask_green(2)+mask_green(4),mask_green(1):mask_green(1)+mask_green(3));
img_jitter_green = img_jitter(mask_green(2):mask_green(2)+mask_green(4),mask_green(1):mask_green(1)+mask_green(3));
img_BM3D_green = img_BM3D(mask_green(2):mask_green(2)+mask_green(4),mask_green(1):mask_green(1)+mask_green(3));
img_unet_green = img_unet(mask_green(2):mask_green(2)+mask_green(4),mask_green(1):mask_green(1)+mask_green(3));
img_noisy_green = img_noisy(mask_green(2):mask_green(2)+mask_green(4),mask_green(1):mask_green(1)+mask_green(3));

img_domain_yellow = img_domain(mask_yellow(2):mask_yellow(2)+mask_yellow(4),mask_yellow(1):mask_yellow(1)+mask_yellow(3));
img_jitter_yellow = img_jitter(mask_yellow(2):mask_yellow(2)+mask_yellow(4),mask_yellow(1):mask_yellow(1)+mask_yellow(3));
img_BM3D_yellow = img_BM3D(mask_yellow(2):mask_yellow(2)+mask_yellow(4),mask_yellow(1):mask_yellow(1)+mask_yellow(3));
img_unet_yellow = img_unet(mask_yellow(2):mask_yellow(2)+mask_yellow(4),mask_yellow(1):mask_yellow(1)+mask_yellow(3));
img_noisy_yellow = img_noisy(mask_yellow(2):mask_yellow(2)+mask_yellow(4),mask_yellow(1):mask_yellow(1)+mask_yellow(3));


%% speckle contrast
mean_domain = mean(mean(img_domain_red));
mean_jitter = mean(mean(img_jitter_red));
mean_BM3D = mean(mean(img_BM3D_red));
mean_unet = mean(mean(img_unet_red));
mean_noisy = mean(mean(img_noisy_red));

std_domain = std2(img_domain_red);
std_jitter = std2(img_jitter_red);
std_BM3D = std2(img_BM3D_red);
std_unet = std2(img_unet_red);
std_noisy = std2(img_noisy_red);

c_domain = std_domain/mean_domain;
c_jitter = std_jitter/mean_jitter;
c_BM3D = std_BM3D/mean_BM3D;
c_unet = std_unet/mean_unet;
c_noisy = std_noisy/mean_noisy;

mean_domain_yellow = mean(img_domain_yellow(:));
mean_jitter_yellow = mean(img_jitter_yellow(:));
mean_BM3D_yellow = mean(img_BM3D_yellow(:));
mean_unet_yellow = mean(img_unet_yellow(:));
mean_original_letter = mean(img_noisy_yellow(:));

c_domain_ratio = c_domain/c_noisy
c_jitter_ratio = c_jitter/c_noisy
c_BM3D_ratio = c_BM3D/c_noisy
c_unet_ratio = c_unet/c_noisy

%% Figure Images

clims_domain = [min(img_domain(:)) max(img_domain(:))];
clims_jitter = [min(img_jitter(:)) max(img_jitter(:))];
clims_BM3D = [min(img_BM3D(:)) max(img_BM3D(:))];
clims_unet = [min(img_unet(:)) max(img_unet(:))];
clims_noisy = [min(img_noisy(:)) max(img_noisy(:))];

% Figure 1 spatial-domain
f1 = figure('Position',[20 450 825 530]);
subplot(2,3,[1,2,4,5])
imshow(img_domain, clims_domain)
% colorbar('AxisLocation','in','Location','westoutside','FontSize',12)

ax = gca;
ax.Units = 'pixels';
ax.Position = [21 10 512 512];
hold on
rectangle('Position', mask_red, 'EdgeColor', 'r', 'LineWidth', 2)
rectangle('Position', mask_green, 'EdgeColor', 'g', 'LineWidth', 2)
rectangle('Position', mask_yellow, 'EdgeColor', 'y', 'LineWidth', 2)

subplot(2,3,3)
imshow(img_domain_green, clims_domain)
rectangle('Position', [1 1 mask_green(3) mask_green(4)], 'EdgeColor', 'g', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 270 252 252];

subplot(2,3,6)
imshow(img_domain_red, clims_domain)
rectangle('Position', [1 1 mask_red(3) mask_red(4)], 'EdgeColor', 'r', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 10 252 252];

% Figure 2 spatial-jitter
f2 = figure('Position',[20 10 825 530]);
subplot(2,3,[1,2,4,5])
imshow(img_jitter, clims_jitter)
% colorbar('AxisLocation','in','Location','westoutside','FontSize',12)

ax = gca;
ax.Units = 'pixels';
ax.Position = [21 10 512 512];
hold on
rectangle('Position', mask_red, 'EdgeColor', 'r', 'LineWidth', 2)
rectangle('Position', mask_green, 'EdgeColor', 'g', 'LineWidth', 2)
rectangle('Position', mask_yellow, 'EdgeColor', 'y', 'LineWidth', 2)

subplot(2,3,3)
imshow(img_jitter_green, clims_jitter)
rectangle('Position', [1 1 mask_green(3) mask_green(4)], 'EdgeColor', 'g', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 270 252 252];

subplot(2,3,6)
imshow(img_jitter_red, clims_jitter)
rectangle('Position', [1 1 mask_red(3) mask_red(4)], 'EdgeColor', 'r', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 10 252 252];

% Figure 3 BM3D
f3 = figure('Position',[540 450 825 530]);
subplot(2,3,[1,2,4,5])
imshow(img_BM3D, clims_BM3D)
% colorbar('AxisLocation','in','Location','westoutside','FontSize',12)

ax = gca;
ax.Units = 'pixels';
ax.Position = [21 10 512 512];
hold on
rectangle('Position', mask_red, 'EdgeColor', 'r', 'LineWidth', 2)
rectangle('Position', mask_green, 'EdgeColor', 'g', 'LineWidth', 2)
rectangle('Position', mask_yellow, 'EdgeColor', 'y', 'LineWidth', 2)

subplot(2,3,3)
imshow(img_BM3D_green, clims_BM3D)
rectangle('Position', [1 1 mask_green(3) mask_green(4)], 'EdgeColor', 'g', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 270 252 252];

subplot(2,3,6)
imshow(img_BM3D_red, clims_BM3D)
rectangle('Position', [1 1 mask_red(3) mask_red(4)], 'EdgeColor', 'r', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 10 252 252];

% Figure 4 U-Net
f4 = figure('Position',[540 10 825 530]);
subplot(2,3,[1,2,4,5])
imshow(img_unet, clims_unet)
% colorbar('AxisLocation','in','Location','westoutside','FontSize',12)

ax = gca;
ax.Units = 'pixels';
ax.Position = [21 10 512 512];
hold on
rectangle('Position', mask_red, 'EdgeColor', 'r', 'LineWidth', 2)
rectangle('Position', mask_green, 'EdgeColor', 'g', 'LineWidth', 2)
rectangle('Position', mask_yellow, 'EdgeColor', 'y', 'LineWidth', 2)

subplot(2,3,3)
imshow(img_unet_green, clims_unet)
rectangle('Position', [1 1 mask_green(3) mask_green(4)], 'EdgeColor', 'g', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 270 252 252];

subplot(2,3,6)
imshow(img_unet_red, clims_unet)
rectangle('Position', [1 1 mask_red(3) mask_red(4)], 'EdgeColor', 'r', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 10 252 252];

% Figure 5 Noisy
f5 = figure('Position',[1060 450 825 530]);
subplot(2,3,[1,2,4,5])
imshow(img_noisy, clims_noisy)
% colorbar('AxisLocation','in','Location','westoutside','FontSize',12)

ax = gca;
ax.Units = 'pixels';
ax.Position = [21 10 512 512];
hold on
rectangle('Position', mask_red, 'EdgeColor', 'r', 'LineWidth', 2)
rectangle('Position', mask_green, 'EdgeColor', 'g', 'LineWidth', 2)
rectangle('Position', mask_yellow, 'EdgeColor', 'y', 'LineWidth', 2)

subplot(2,3,3)
imshow(img_noisy_green, clims_noisy)
rectangle('Position', [1 1 mask_green(3) mask_green(4)], 'EdgeColor', 'g', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 270 252 252];

subplot(2,3,6)
imshow(img_noisy_red, clims_noisy)
rectangle('Position', [1 1 mask_red(3) mask_red(4)], 'EdgeColor', 'r', 'LineWidth', 2)

ax = gca;
ax.Units = 'pixels';
ax.Position = [553 10 252 252];

%% Relative deviation
img_domain_rd = (img_domain_red - mean_domain)/mean_domain;
img_jitter_rd = (img_jitter_red - mean_jitter)/mean_jitter;
img_BM3D_rd = (img_BM3D_red - mean_BM3D)/mean_BM3D;
img_unet_rd = (img_unet_red - mean_unet)/mean_unet;
img_noisy_rd = (img_noisy_red - mean_noisy)/mean_noisy;

img_domain_letter_rd = (img_domain_yellow - mean_domain_yellow)/mean_domain_yellow;
img_jitter_letter_rd = (img_jitter_yellow - mean_jitter_yellow)/mean_jitter_yellow;
img_BM3D_letter_rd = (img_BM3D_yellow - mean_BM3D_yellow)/mean_BM3D_yellow;
img_unet_letter_rd = (img_unet_yellow - mean_unet_yellow)/mean_unet_yellow;
img_noisy_letter_rd = (img_noisy_yellow - mean_original_letter)/mean_original_letter;

%% Figure Relative deviation
f6 = figure;
surf(img_domain_rd,'EdgeColor','none')
% mesh(img_noisy_rd)
xlim([0 mask_red(3)+1])
ylim([0 mask_red(3)+1])
zlim([-1 3])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-3 3])

f7 = figure;
surf(img_jitter_rd,'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_red(3)+1])
ylim([0 mask_red(3)+1])
zlim([-1 3])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-3 3])

f8 = figure;
surf(img_BM3D_rd,'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_red(3)+1])
ylim([0 mask_red(3)+1])
zlim([-1 3])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-3 3])

f9 = figure;
surf(img_unet_rd,'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_red(3)+1])
ylim([0 mask_red(3)+1])
zlim([-1 3])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-3 3])

f10 = figure;
surf(img_noisy_rd,'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_red(3)+1])
ylim([0 mask_red(3)+1])
zlim([-1 3])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-3 3])

%% Figure Letter

f11 = figure;
surf(flipud(img_domain_letter_rd),'EdgeColor','none')
% mesh(img_noisy_rd)
xlim([0 mask_yellow(3)+1])
ylim([0 mask_yellow(4)+1])
zlim([-2 2])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-1 1.5])
pbaspect([2 6 0.7])

f12 = figure;
surf(flipud(img_jitter_letter_rd),'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_yellow(3)+1])
ylim([0 mask_yellow(4)+1])
zlim([-2 2])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-1 1.5])
pbaspect([2 6 0.7])

f13 = figure;
surf(flipud(img_BM3D_letter_rd),'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_yellow(3)+1])
ylim([0 mask_yellow(4)+1])
zlim([-2 2])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-1 1.5])
pbaspect([2 6 0.7])

f14 = figure;
surf(flipud(img_unet_letter_rd),'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_yellow(3)+1])
ylim([0 mask_yellow(4)+1])
zlim([-2 2])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-1 1.5])
pbaspect([2 6 0.7])

f15 = figure;
surf(flipud(img_noisy_letter_rd),'EdgeColor','none')
% mesh(img_denoise_rd)
xlim([0 mask_yellow(3)+1])
ylim([0 mask_yellow(4)+1])
zlim([-2 2])
xlabel('pixels', 'FontSize', 13)
ylabel('pixels', 'FontSize', 13)
zlabel('R.D', 'FontSize', 13)
ax = gca;
ax.FontSize = 13;
colorbar('FontSize',13)
caxis([-1 1.5])
pbaspect([2 6 0.7])

%% Save figure as eps file
if image_save == 1
print(f1,'-depsc2','-painters','PCB_dynami_mask.eps')
print(f2,'-depsc2','-painters','PCB_spatial_jitter.eps')
print(f3,'-depsc2','-painters','PCB_BM3D.eps')
print(f4,'-depsc2','-painters','PCB_Unet.eps')
print(f5,'-depsc2','-painters','PCB_noisy.eps')

print(f11,'-depsc2','-painters','PCB_RD_dynami_mask.eps')
print(f12,'-depsc2','-painters','PCB_RD_spatial_jitter.eps')
print(f13,'-depsc2','-painters','PCB_RD_BM3D.eps')
print(f14,'-depsc2','-painters','PCB_RD_Unet.eps')
print(f15,'-depsc2','-painters','PCB_RD_noisy.eps')
end