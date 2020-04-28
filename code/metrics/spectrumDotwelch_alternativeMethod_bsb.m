%% 4th April For Ajay S: the code which i use to generate power spectra for model outputs
% 6th April testing with the data that Ajay sent yesterday. Imported the
% top part of his code to call in the 4 .csv files containing spike count
% data.

close all
clear all

TotalDuration = 2000;
tl = 200;
th = 1800;
gs = [0.1, 0.2, 0.3, 0.4];
ylims = [0.2, 0.4, 0.6, 0.9];
figure(1);
sgtitle('PSD: \lambda=5-14 and g=0.1-0.4');
for i=1:4
    %% read csv files
    Data1 = csvread(sprintf('../../experiments/2k/ret_tcr/csv/TCR_spikes_%d_5.csv',i),0,1);
    spikeTimes1 = Data1(1:1:end);
    spikeTimes1 = spikeTimes1(spikeTimes1 > tl);
    spikeTimes1 = spikeTimes1(spikeTimes1 < th);
    Data2 = csvread(sprintf('../../experiments/2k/ret_tcr/csv/TCR_spikes_%d_8.csv',i),0,1);
    spikeTimes2 = Data2(1:1:end);
    spikeTimes2 = spikeTimes2(spikeTimes2 > tl);
    spikeTimes2 = spikeTimes2(spikeTimes2 < th);
    Data3 = csvread(sprintf('../../experiments/2k/ret_tcr/csv/TCR_spikes_%d_11.csv',i),0,1);
    spikeTimes3 = Data3(1:1:end);
    spikeTimes3 = spikeTimes3(spikeTimes3 > tl);
    spikeTimes3 = spikeTimes3(spikeTimes3 < th);
    Data4 = csvread(sprintf('../../experiments/2k/ret_tcr/csv/TCR_spikes_%d_14.csv',i),0,1);
    spikeTimes4 = Data4(1:1:end);
    spikeTimes4 = spikeTimes4(spikeTimes4 > tl);
    spikeTimes4 = spikeTimes4(spikeTimes4 < th);

    %% labels that will be used in plots
    name1 = sprintf('g=%.1f \\lambda=5',gs(i));
    name2 = sprintf('g=%.1f \\lambda=8',gs(i));
    name3 = sprintf('g=%.1f \\lambda=11',gs(i));
    name4 = sprintf('g=%.1f \\lambda=14',gs(i));

    %% population spk histogram
    spk_count1 = hist(spikeTimes1,0:TotalDuration);%for PSD calculation
    spk_count2 = hist(spikeTimes2,0:TotalDuration);%for PSD calculation
    spk_count3 = hist(spikeTimes3,0:TotalDuration);%for PSD calculation
    spk_count4 = hist(spikeTimes4,0:TotalDuration);%for PSD calculation

    spike_count_mat=[spk_count1; spk_count2; spk_count3; spk_count4];

    %% PARAMETERS FOR COMPUTING THE POWER SPECTRA
    Fs = 1000;  % sampling frequency for downsampling
    NFFT = 8*Fs;  % number of FFT points (multiple of sampling frequency here)
    window_type = 'hamming';  % type of window used to extract time segments
    segment_length = (1/4)*Fs;  % length of time segments (in samples) into which the signal should be divided for spectral analysis (separate analyses are done on each segment and then averaged)
    overlap_percent = 50;  % percentage overlap between successive time segments
    normalised = 0;  % we specify the sampling frequency and so do not wish the sampling frequency in subsequent filtering to be automatically normalised
    hp = spectrum.welch(window_type,segment_length,overlap_percent);  % object for spectral analysis using Welch's periodogram method

    %% PARAMETERS FOR BAND PASS FILTERING
    Fc1 = 1;  % low cutoff frequency
    Fc2 = 100;  % high cutoff frequency
    N_filt = 10;  % filter order
    h  = fdesign.bandpass('N,F3dB1,F3dB2', N_filt, Fc1, Fc2, Fs);  % bandpass filter designer
    hd = design(h, 'butter');  % bandpass filter itself
    [B_filt,A_filt] = sos2tf(hd.sosMatrix,hd.Scalevalues);  % transfer function related to filter

    %% Compute and Plot the PSD of the filtered histogram
    subplot(2,2,i);
    style = ['-', "--"];
    color = ['r', 'g', 'b', 'm'];
    for j = 1:size(spike_count_mat,1)
        spikecount_tcr=spike_count_mat(j,:);

        % Filter Data and Get Power Spectral Density
        filt_data_tcr = filtfilt(B_filt,A_filt,spikecount_tcr);%% 
        hpopts_tcr = psdopts(hp,filt_data_tcr);  % get options for the power spectral density object 
        set(hpopts_tcr,'Fs',Fs,'NFFT',NFFT,'Normalized',normalised);  % set/update options for the power spectral density object 
        hpsd_tcr = psd(hp,filt_data_tcr,hpopts_tcr);  % get power spectral density object for filtered data using spectral analysis object and options
        spec_power_tcr = hpsd_tcr.Data';  % get actual power spectral density values from object (NFFT/2 + 1 values in total)
        spec_freq_tcr = hpsd_tcr.Frequencies;  % power spectral density frequencies (one corresponding to each power spectral density value; same for all objects so only one needed)

        plot(spec_freq_tcr,spec_power_tcr,style(mod(j,2)+1)+color(j), 'LineWidth',1);
        hold on
        xlim([1, 50])
        ylim([0, ylims(i)]);
        xlabel('Frequency (Hz)');
        ylabel('PSD');
    end
    legend(name1, name2, name3, name4);
%legend(ax2,{name3, name4});
end
%% CREATE A CONCATENATED MATRIX AND PERFORM STFT - not working I think....
% spike_count_mat_2=normalize(spk_count1);%[spk_count1 spk_count2 spk_count3 spk_count4];
% 
% [fr, vismat]=fun_stft(spike_count_mat_2, 1, 50, 0,0,0);
% figure, imagesc([],fr((4*Fc1+1):(4*Fc2+1)),vismat((4*Fc1+1):(4*Fc2+1),2:end));
% xlabel('Time windows','Fontsize',14);
% ylabel('frequency(Hz)','Fontsize',14);
% axis('xy')
% set(gca,'Fontsize',12),colorbar
% % ylim([1 127])
% title('input frequency is  Hz')