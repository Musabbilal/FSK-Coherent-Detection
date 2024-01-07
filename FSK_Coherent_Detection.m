% On-Off signaling
f = 1; 
T = 1; 
Fs = 100; 
t = 0:1/Fs:T-1/Fs; 

%Source signal
bits = [0 0 0 1 1 0 0 0];


digit = []
signal = []

for bit = bits
    
    if bit == 0
        signal = [signal zeros(1,length(t))];
    else
        signal = [signal ones(1,length(t))];
    end
    digit = [digit signal];
end

figure;
plot(signal, 'LineWidth',2.5);
axis([0 800 -0.5 1.5]);
title('OOK Signal');
xlabel('Time');
ylabel('Amplitude');

% FSK signal
signal = [];
f1 = 1;
f2 = 5;
% Generate the FSK signal
for bit = bits
    if bit == 0
        signal = [signal cos(2*pi*f1*t)];
    else
        signal = [signal cos(2*pi*f2*t)];
    end
end

% Plot the signal
figure;
plot(signal);
axis([0 800 -1.5 1.5]);
title('FSK Signal');
xlabel('Time');
ylabel('Amplitude');



    prompt = "choose your method , 1 (normal), 2(FFT) ,3 (PLL), 4 (Hilbert), 5 (Goertzel)? ";
    userInput = input(prompt)
    

if(userInput == 1) 
    %FSK coherent detection
received_signal = signal_noisy; % your received FSK signal here

demodulated_signal = [];

% Demodulate the FSK signal
for i = 1:length(t):length(received_signal)
    % Extract one bit period of the received signal
    segment = received_signal(i:i+length(t)-1);
    
    % Mixers
    mixed_signal1 = segment .* cos(2*pi*f1*t);
    mixed_signal2 = segment .* cos(2*pi*f2*t);
 
    % Low-pass filters
    filtered_signal1 = lowpass(mixed_signal1, 1/(2*T), Fs);
    filtered_signal2 = lowpass(mixed_signal2, 1/(2*T), Fs);
    
    % Make a decision based on the energy of the filtered signals
    if sum(filtered_signal1.^2) > sum(filtered_signal2.^2)
        demodulated_signal = [demodulated_signal 0];
    else
        demodulated_signal = [demodulated_signal 1];
    end
end
figure 
subplot (2,1,1)
plot (t,mixed_signal1, t, filtered_signal1)
legend ('mixed', 'filtered')
title ('Signal for 0')
subplot (2,1,2)
plot (t,mixed_signal2,t,filtered_signal2)
legend ('mixed','filtered')
title ('Signal for 1')


disp('Received bits:');
disp(demodulated_signal);

%Testing
if isequal (demodulated_signal,bits)
    disp ('AA')

else
    disp ('FF')
end

% FFT
elseif (userInput == 2)  
N = length(signal);
y = fft(signal, N);

% Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2
P2 = abs(y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);


f = Fs*(0:(N/2))/N;

% Find the peak frequencies
[~,locs] = findpeaks(P1,'MinPeakHeight',0.2); % Lower the 'MinPeakHeight'


if length(locs) < 2
    fprintf('Not enough peaks found. Try lowering the MinPeakHeight.\n');
else
    
    fprintf('Detected frequencies: %f Hz, %f Hz\n', f(locs));
end
 
    fprintf('Detected frequencies: %f Hz, %f Hz\n', f(locs));

    % Generate filters
    filter1 = cos(2*pi*f(locs(1))*t);
    filter2 = cos(2*pi*f(locs(2))*t);

    % Apply the filters
    output1 = conv(signal, filter1, 'same');
    output2 = conv(signal, filter2, 'same');

    % Make a decision based on the energy in the filter outputs
    output = abs(output1) < abs(output2);


    figure;
    plot(output);
    title('Output of Filter-based FSK Demodulator');
    xlabel('Time');
    ylabel('Amplitude');

    output(1:100:800)



received_signal = signal;

% Demodulate the FSK signal
demodulated_signal = [];
for i = 1:length(t):length(received_signal)
    % Extract one bit period of the received signal
    segment = received_signal(i:i+length(t)-1);

    % Mixers
    mixed_signal1 = segment .* cos(2*pi*f(locs(1))*t);
    mixed_signal2 = segment .* cos(2*pi*f(locs(2))*t);

    % Low-pass filters
    filtered_signal1 = lowpass(mixed_signal1, 1/(2*T), Fs);
    filtered_signal2 = lowpass(mixed_signal2, 1/(2*T), Fs);




    % Make a decision based on the energy of the filtered signals
    if sum(filtered_signal1.^2) > sum(filtered_signal2.^2)
        demodulated_signal = [demodulated_signal 0];
    else
        demodulated_signal = [demodulated_signal 1];
    end
end
    figure
    subplot (2,1,1);
    plot (filtered_signal1);
    title ('Filtered signal for 1')
    subplot(2,1,2)
    plot (filtered_signal2)
    title ('Filtered signal for 0')

  disp('Received bits:');
  disp(demodulated_signal)


    %Testing
  if isequal (demodulated_signal,bits)
    disp ('AA')

else disp ('Conditional pass')

end


 %PLL method
elseif (userInput == 3)
   
    input_signal = signal_noisy;
% Initialize PLL
phase_difference = 0; 
vco_output = zeros(length(signal), 1); 
vco_input = 0; 


mixed_signal = zeros(length(signal), 1); 
filtered_signal = zeros(length(signal), 1); 
demodulated_signal = [];
% Define parameters
Fs = 100; 
T = 1/Fs; 
L = length(signal); 
t = (0:L-1)*T; 

% Define PLL parameters
Kd = 1; 
Kv = 1; 
N = 1; 

% Initialize PLL
phase_difference = 0; 
vco_output = zeros(length(signal), 1);
vco_input = 0; 
tol = .02;

% Loop through each sample
mixed_signal = zeros(length(signal), 1); 
filtered_signal = zeros(length(signal), 1); 
demodulated_signal = [];

%PLL loop
for i = 3:length(signal)-1
    % Phase detector
    phase_difference = input_signal(i) - vco_output(i);

    % Low-pass filter
    vco_input = Kd * phase_difference;

    % VCO
    if i < L
        vco_output(i+1) = sin(Kv * vco_input);
        mixed_signal(i) = input_signal(i) .* vco_output(i+1);
        filtered_signal(i) = lowpass(mixed_signal(i), 1/(2*T), Fs);
    end
end

vco_output = 2*vco_output;
% Plot input signal and Filtered signal
figure;
subplot(2,1,1);
plot(input_signal);
axis([0 800 -1.5 1.5]);
title('Input Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(filtered_signal);
axis([0 800 -1.5 1.5]);
title('Filtered signal');
xlabel('Time (s)');
ylabel('Amplitude');
demodulated_signal = [];
received_signal = input_signal;
f = 1; 
T = 1; 
Fs = 100; 
t = 0:1/Fs:T-1/Fs; 
for i = 1:length(t):length(received_signal)
    segment = received_signal(i:i+length(t)-1);
    
    % Mixers
    mixed_signal1 = segment .* cos(2*pi*f1*t);
    mixed_signal2 = segment .* cos(2*pi*f2*t);
 
    % Low-pass filters
    filtered_signal1 = lowpass(mixed_signal1, 1/(2*T), Fs);
    filtered_signal2 = lowpass(mixed_signal2, 1/(2*T), Fs);
    
    % Make a decision based on the energy of the filtered signals
    if sum(filtered_signal1.^2) > sum(filtered_signal2.^2)
        demodulated_signal = [demodulated_signal 0];
    else
        demodulated_signal = [demodulated_signal 1];
    end
end
disp('Received bits:');
disp(demodulated_signal);

  %Testing
if isequal (demodulated_signal,bits)
    disp ('AA')

else disp ('FF')
end

% Hilbert Transform
elseif (userInput == 4)
hilbert_transform = hilbert(signal);

% Instantaneous Phase
inst_phase = unwrap(angle(hilbert_transform));

% Instantaneous Frequency
inst_freq = diff(inst_phase)/(2*pi*(t(2)-t(1)));
% Compute histogram
[counts, edges] = histcounts(inst_freq);

% Find the two most common frequencies, which should correspond to f1 and f2
[freqs_val, freqs_edges] = histcounts(inst_freq);
[~, sorted_freqs_indices] = sort(freqs_val, 'descend');

%The Extracted freqs
f1_est = freqs_edges(sorted_freqs_indices(1))
f2_est = freqs_edges(sorted_freqs_indices(2))*10

% Plot the Instantaneous Frequency
figure;
plot( inst_freq);
title('Instantaneous Frequency using Hilbert Transform');
xlabel('Time');
ylabel('Frequency');

demodulated_signal = [];

for i = 1:length(t):length(signal)
    % Extract one bit period of the received signal
    segment = signal(i:i+length(t)-1);
    
    % Mixers
    mixed_signal1 = segment .* cos(2*pi*f1_est*t);
    mixed_signal2 = segment .* cos(2*pi*f2_est*t);
 
    % Low-pass filters
    filtered_signal1 = lowpass(mixed_signal1, 1/(2*T), Fs);
    filtered_signal2 = lowpass(mixed_signal2, 1/(2*T), Fs);
    
    % Make a decision based on the energy of the filtered signals
    if sum(filtered_signal1.^2) > sum(filtered_signal2.^2)
        demodulated_signal = [demodulated_signal 0];
    else
        demodulated_signal = [demodulated_signal 1];
    end
end
disp('Received bits:');
disp (demodulated_signal)

  %Testing
if isequal (output_bits,bits)
    disp ('AA')

else disp ('Conditional pass')

end

% Goertzel Algorithm
elseif (userInput == 5)
N = length(t);
k1 = 0.5 + f1 * N / Fs;
k2 = 0.5 + f2 * N / Fs;
coeff1 = 2*cos(2*pi*k1/N);
coeff2 = 2*cos(2*pi*k2/N);

% BFSK Demodulation
received_bits = [];
for i = 1:length(bits)
    s_prev1 = 0;
    s_prev2 = 0;
    s_prev3 = 0;
    s_prev4 = 0;
    for j = 1:N
        s = signal((i-1)*N+j) + coeff1 * s_prev1 - s_prev2;
        s_prev2 = s_prev1;
        s_prev1 = s;
        s = signal((i-1)*N+j) + coeff2 * s_prev3 - s_prev4;
        s_prev4 = s_prev3;
        s_prev3 = s;
    end
    power1 = s_prev2*s_prev2 + s_prev1*s_prev1 - coeff1*s_prev1*s_prev2;
    power2 = s_prev4*s_prev4 + s_prev3*s_prev3 - coeff2*s_prev3*s_prev4;
    if power1 > power2
        received_bits = [received_bits 0];
    else
        received_bits = [received_bits 1];
    end
end
figure;
plot(power1_values, 'r');
hold on;
plot(power2_values, 'b');
title('Power Spectrum');
xlabel('Bit Index');
ylabel('Power');
legend('Power at 0', 'Power at 1');

disp('Received bits:');
disp(received_bits);

  %Testing
if isequal (output_bits,bits)
    disp ('AA')

else disp ('FF')

end


% Showing noise effects (FFT)
elseif (userInput == 10)  
N = length(signal_noisy);
y = fft(signal_noisy, N);

% Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2
P2 = abs(y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);


f = Fs*(0:(N/2))/N;

% Find the peak frequencies
[~,locs] = findpeaks(P1,'MinPeakHeight',0.2); % Lower the 'MinPeakHeight'


if length(locs) < 2
    fprintf('Not enough peaks found. Try lowering the MinPeakHeight.\n');
else
    
    fprintf('Detected frequencies: %f Hz, %f Hz\n', f(locs));
end
 
    fprintf('Detected frequencies: %f Hz, %f Hz\n', f(locs));

    % Generate filters
    filter1 = cos(2*pi*f(locs(1))*t);
    filter2 = cos(2*pi*f(locs(2))*t);

    % Apply the filters
    output1 = conv(signal_noisy, filter1, 'same');
    output2 = conv(signal_noisy, filter2, 'same');

    % Make a decision based on the energy in the filter outputs
    output = abs(output1) < abs(output2);


    figure;
    plot(output);
    title('Output');
    xlabel('Time');
    ylabel('Amplitude');

    output(1:100:800)
     
    % Reshape each row represents one bit
    output_matrix = reshape(output, [length(t), length(bits)]);
    mean_output = mean(output_matrix, 1);
    output_bits = mean_output > 0.5;

  %Testing
if isequal (output_bits,bits)
    disp ('AA')

else disp ('Conditional pass')

end


else disp ('input a number from 1--3')
end
