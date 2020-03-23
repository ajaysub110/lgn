clear all
close all
clc

projs = ["ret_tcr","trn_tcr","in_tcr","ret_in","tcr_trn"];
projs_name = ["ret-tcr","trn-tcr","in-tcr","ret-in","tcr-trn"];
Pxx_C = {}; % cell to store PSD values
TotalDuration = 1000;

%pwelch parameters
fs = 1000; %inverse 0.1ms time step
nfft = 256; noverlap = nfft/2; wind = hamming(nfft);
L = 1;
dT = 100;

% prepare figure
fig1 = figure(1);
set(fig1,'Position',[50 100 600 300]);
sgtitle('Concatenated PSD colormaps');

% calculate total psd for each projection
for j=1:length(projs) % for each proj: tcr_trn, ret_tcr ...
    spikeTimes = [];
    for i=0:9 % for each conductance value
        path = strcat('../../experiments/synchrony/sync/',projs(j),'/TCR_spikes_',string(i),'.csv');
        Data = csvread(path,0,1);
        spikeTimes = cat(1,spikeTimes,Data(1:1:end)); % concatenate spike times
    end
    spk_count = hist(spikeTimes,0:TotalDuration); % spike count histogram
    count = 1;
    t0 = L*nfft/2+1;

    % calculate windowed PSD
    while (t0+L*nfft/2) < TotalDuration
        [Pxx(:,count),F] = pwelch(spk_count(t0-L*nfft/2:t0+L*nfft/2-1),wind,noverlap,nfft,fs,'psd');
        Pxx(:,count) = Pxx(:,count)/sum(Pxx(:,count));
        t_fft(count) = t0;
        count = count+1;
        t0 = t0+dT;
    end

    Pxx_C{j} = Pxx; % add PSD value to cell
    
    % Plot current projection's PSD colormap
    subplot(3,2,j);
    p = pcolor(t_fft,F,Pxx); set(p,'LineStyle','none');
    xlim([t_fft(1) t_fft(end)]); %ylim([1 50]);
    xlabel('Time(ms)'); ylabel('Frequency(Hz)');
    title('PSD','FontSize',16); title(projs_name(j));
    ylim([0 50]);
    colorbar;
end
