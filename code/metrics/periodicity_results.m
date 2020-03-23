clear all
close all
clc

% read csv files
Data1 = csvread('../../experiments/synchrony/sync/ret_tcr/TCR_spikes_0.csv',0,1);
spikeTimes1 = Data1(1:1:end);
Data2 = csvread('../../experiments/synchrony/sync/ret_tcr/TCR_spikes_3.csv',0,1);
spikeTimes2 = Data2(1:1:end);
Data3 = csvread('../../experiments/synchrony/sync/ret_tcr/TCR_spikes_6.csv',0,1);
spikeTimes3 = Data3(1:1:end);
Data4 = csvread('../../experiments/synchrony/sync/ret_tcr/TCR_spikes_9.csv',0,1);
spikeTimes4 = Data4(1:1:end);

% labels that will be used in plots
name1 = 'tcr-g0pt1';
name2 = 'tcr-g0pt4';
name3 = 'tcr-g0pt7';
name4 = 'tcr-g1';

%population spk histogram
TotalDuration = 1000;
spk_count1 = hist(spikeTimes1,0:TotalDuration);%for PSD calculation
spk_count2 = hist(spikeTimes2,0:TotalDuration);%for PSD calculation
spk_count3 = hist(spikeTimes3,0:TotalDuration);%for PSD calculation
spk_count4 = hist(spikeTimes4,0:TotalDuration);%for PSD calculation

figure(4);
plot(spk_count1, 'r');
hold on
plot(spk_count2, 'b');
hold on
plot(spk_count3, 'g');
hold on
plot(spk_count4, 'y');
title('Spike count histogram plot');
ylabel('Spike count');
legend(name1,name2,name3,name4);
xlabel('Time (ms)');
xlim([0,1000]);

% smoothen spike count histogram by applying gaussian filter
spk_count_lowfreq1 = filtfilt(fspecial('gaussian',[1 100],20), 1, spk_count1); %time averaged
spk_count_lowfreq2 = filtfilt(fspecial('gaussian',[1 100],20), 1, spk_count2); %time averaged
spk_count_lowfreq3 = filtfilt(fspecial('gaussian',[1 100],20), 1, spk_count3); %time averaged
spk_count_lowfreq4 = filtfilt(fspecial('gaussian',[1 100],20), 1, spk_count4); %time averaged

%pwelch (for PSD) parameters
fs = 1000; %inverse 0.1ms time step
nfft = 256; noverlap = nfft/2; wind = hamming(nfft);
count = 1; L = 1;
dT = 100; t0 = L*nfft/2+1;

% Apply windowed filter and calculate pwelch for each
while (t0+L*nfft/2) < TotalDuration
    [Pxx1(:,count),F] = pwelch(spk_count1(t0-L*nfft/2:t0+L*nfft/2-1),wind,noverlap,nfft,fs,'psd');
    Pxx1(:,count) = Pxx1(:,count)/sum(Pxx1(:,count));
    [Pxx2(:,count),F] = pwelch(spk_count2(t0-L*nfft/2:t0+L*nfft/2-1),wind,noverlap,nfft,fs,'psd');
    Pxx2(:,count) = Pxx2(:,count)/sum(Pxx2(:,count));
    [Pxx3(:,count),F] = pwelch(spk_count3(t0-L*nfft/2:t0+L*nfft/2-1),wind,noverlap,nfft,fs,'psd');
    Pxx3(:,count) = Pxx3(:,count)/sum(Pxx3(:,count));
    [Pxx4(:,count),F] = pwelch(spk_count4(t0-L*nfft/2:t0+L*nfft/2-1),wind,noverlap,nfft,fs,'psd');
    Pxx4(:,count) = Pxx4(:,count)/sum(Pxx4(:,count));
    t_fft(count) = t0;
    count = count+1;
    t0 = t0+dT;
end

% PSD vs frequency plot
figure(5);
sgtitle('PSD vs frequency plot');
subplot(221);
for i=1:8
    plot(F,Pxx1(:,i));
    title(name1)
    xlim([0,100]);
    hold on
end
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
subplot(222);
for i=1:8
    plot(F,Pxx2(:,i));
    title(name2)
    xlim([0,100]);
    hold on
end
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
subplot(223);
for i=1:8
    plot(F,Pxx3(:,i));
    title(name3)
    xlim([0,100]);
    hold on
end
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
subplot(224);
for i=1:8
    plot(F,Pxx4(:,i));
    title(name4)
    xlim([0,100]);
    hold on
end
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');

% Calculate correlations of PSD with frequency and amplitude
fmin = 20; fmax = 80;
for n = 1:length(t_fft)
    temp = find(Pxx1(F>fmin,n) == max(Pxx1(F>fmin&F<fmax,n)),1);
    if length(temp)
        fmax1(n) = temp;
    else fmax1(n) = 0;
    end
    temp = find(Pxx2(F>fmin,n) == max(Pxx2(F>fmin&F<fmax,n)),1);
    if length(temp)
        fmax2(n) = temp;
    else fmax2(n) = 0;
    end
    temp = find(Pxx3(F>fmin,n) == max(Pxx3(F>fmin&F<fmax,n)),1);
    if length(temp)
        fmax3(n) = temp;
    else fmax3(n) = 0;
    end
    temp = find(Pxx4(F>fmin,n) == max(Pxx4(F>fmin&F<fmax,n)),1);
    if length(temp)
        fmax4(n) = temp;
    else fmax4(n) = 0;
    end
    Pmax1(n) = max(Pxx1(F>fmin,n));
    Pmax2(n) = max(Pxx2(F>fmin,n));
    Pmax3(n) = max(Pxx3(F>fmin,n));
    Pmax4(n) = max(Pxx4(F>fmin,n));
end
R = corrcoef([Pmax1(:) spk_count_lowfreq1(t_fft)']);  corrP(1) = R(1,2);
R = corrcoef([Pmax2(:) spk_count_lowfreq2(t_fft)']);  corrP(2) = R(1,2);
R = corrcoef([Pmax3(:) spk_count_lowfreq3(t_fft)']);  corrP(3) = R(1,2);
R = corrcoef([Pmax4(:) spk_count_lowfreq4(t_fft)']);  corrP(4) = R(1,2);
R = corrcoef([fmax1(:) spk_count_lowfreq1(t_fft)']);  corrF(1) = R(1,2);
R = corrcoef([fmax2(:) spk_count_lowfreq2(t_fft)']);  corrF(2) = R(1,2);
R = corrcoef([fmax3(:) spk_count_lowfreq3(t_fft)']);  corrF(3) = R(1,2);
R = corrcoef([fmax4(:) spk_count_lowfreq4(t_fft)']);  corrF(4) = R(1,2);

% Plot smoothened no_of_spikes vs time histogram 
fig1 = figure(1);
set(fig1,'Position',[50 100 600 300]);
% subplot(2,1,1);
% plot(firings(:,1),firings(:,2),'.'); xlim([1 TotalDuration]);
% xlabel('Time(ms)'); ylabel('#Unit');
% subplot(2,1,2); hold on;
plot(0:TotalDuration,fs*spk_count_lowfreq1/80,'r'); xlim([0 TotalDuration]);hold on
plot(0:TotalDuration,fs*spk_count_lowfreq2/80,'b'); xlim([0 TotalDuration]);hold on
plot(0:TotalDuration,fs*spk_count_lowfreq3/80,'g'); xlim([0 TotalDuration]);hold on
plot(0:TotalDuration,fs*spk_count_lowfreq4/80,'y'); xlim([0 TotalDuration]);hold on
xlabel('Time(ms)'); ylabel('Population Rate(Hz)');
legend(name1,name2,name3,name4);
title('Filtered spike rate plot');

% PSD color plot (not being used now. please see total_psd.m file) 
fig2 = figure(2);
set(fig2,'Position',[50 100 600 300]);
subplot(221);
p = pcolor(t_fft,F,Pxx1); set(p,'LineStyle','none');
xlim([t_fft(1) t_fft(end)]); %ylim([1 50]);
xlabel('Time(ms)'); ylabel('Frequency(Hz)');
title('PSD','FontSize',16); title(name1);
ylim([0 50]);
colorbar;
subplot(222);
p = pcolor(t_fft,F,Pxx2); set(p,'LineStyle','none');
% xlim([t_fft(1) t_fft(end)]); ylim([1 400]);
xlabel('Time(ms)'); ylabel('Frequency(Hz)');
title('PSD','FontSize',16); title(name2);
ylim([0 50]);
colorbar;
subplot(223);
p = pcolor(t_fft,F,Pxx3); set(p,'LineStyle','none');
xlim([t_fft(1) t_fft(end)]); ylim([0 50]);
xlabel('Time(ms)'); ylabel('Frequency(Hz)');
title('PSD','FontSize',16); title(name3);
colorbar;
ylim([0 50]);
subplot(224);
p = pcolor(t_fft,F,Pxx4); set(p,'LineStyle','none');
%xlim([t_fft(1) t_fft(end)]); ylim([0 50]);
xlabel('Time(ms)'); ylabel('Frequency(Hz)');
title('PSD','FontSize',16); title(name4);
colorbar;
ylim([0 50]);
% 
% fig3 = figure(3);
% %
% set(fig3,'Position',[50 100 600 300]);
% h1 = subplot(1,2,1);
% bar(1:3, corrP); xlim([0 4]);
% set(h1,'XTick',[1:3]);
% set(h1,'XTickLabel',{'TCR0', 'TCR1', 'TCR2'});
% title('Power Corr');
% %
% h2 = subplot(1,2,2);
% bar(1:3, corrF); xlim([0 4]);
% set(h2,'XTick',[1:3]);
% set(h2,'XTickLabel',{'TCR0', 'TCR1', 'TCR2'});
% title('Freq Corr');

