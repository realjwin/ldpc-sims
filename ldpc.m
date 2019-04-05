clear all; close all;

%variables
N = 64;
cp = 0; %specifies # of channel taps (taps = cp+1)
num_blocks = 5*N; %this is number of codeblocks sent, must be multiple of N
oversampling = 1;
snrdb = [0, 1, 1.3, 1.6:.1:4];
n_snrdb = length(snrdb);

%LDPC + OFDM variables
%block length: 64800, rate: 1/2
hEnc = comm.LDPCEncoder; 
hDec = comm.LDPCDecoder;
rate = 1/2;
block_size = 64800;
n_cbits = block_size*num_blocks;
n_bits = block_size*num_blocks*rate;
n_ofdm_symbols = n_cbits / (N*log2(4)); %qpsk
bits = zeros(block_size*rate, num_blocks);
cbits = zeros(block_size, num_blocks);

%%%---GENERATE & ENCODE DATA---%%%
%generate and encode data (each column is one ldpc block)
for k = 1:num_blocks,
    bits(:, k) = logical(randi([0 1], block_size*rate, 1));
    cbits(:, k) = step(hEnc, bits(:,k));
end

%%%---GENERATE DFT MATRIX---%%%

%DFT Matrix
w = exp(-2*pi*i/N);
W = zeros(N,N);
for j=0:N-1,
    for k=0:N-1,
        W(j+1,k+1) = w^(j*k)/sqrt(N);
    end
end

%Eigenvalue Generator Matrix
temp_iter = 0:N-1;
w = exp(-2*pi*i.*temp_iter./N);
for k=0:cp,
    Wp(k+1,:) = w.^k;
end

%%%---SNR ITERATIONS---%%%
ber_uncoded = zeros(n_snrdb, 1);
ber_quantized = zeros(n_snrdb, 1);
bler_quantized = zeros(n_snrdb, 1);
ber_soft = zeros(n_snrdb, 1);
bler_soft = zeros(n_snrdb, 1);
ber_hard = zeros(n_snrdb, 1);
bler_hard = zeros(n_snrdb, 1);

for m=1:n_snrdb,
    
    snrdb(m)
    
    %%%---MODULATE DATA---%%%
    %concatenates rows together
    cbits_temp = reshape(cbits, 2, n_cbits/2)';
    %bi2de is little-endian, (i.e. read in reverse, e.g. 01 = 2)
    %assume everything is little-endian!
    tx_samples = bi2de(cbits_temp);

    %qpsk modulation
    %from quad 1 to 4 (counterclockwise): 01 (2), 00 (0), 10 (1), 11 (3)
    tx_symbols = (1/sqrt(2)) * qammod(tx_samples, 4);

    %%%---OFDM---%%%

    %each ofdm symbol is a column of length N
    tx_ofdm_symbols = reshape(tx_symbols, N, n_ofdm_symbols);

    %%%---CHANNEL---%%%

    %channel is Rayleigh(1)
    h = normrnd(0,1,n_ofdm_symbols,cp+1) + i*normrnd(0,1,n_ofdm_symbols,cp+1);
    %normalize h
    h = h ./ sqrt(sum(abs(h).^2, 2));
    
    %%%---EIGENVALUES---%%%
    %each column is N eigenvalues
    lambda = (h*Wp)';
    
    %%%---NOISE---%%%
    
    %generate noise enhancement matrix [ inv(L)*W ]
    %y = HW'x+n, inv(L)*Wy = x + inv(L)*Wn
    
    %add noise (and noise enhancement)
    noise = (1 / sqrt(10^(snrdb(m)/10))) * (1/sqrt(2)) * ...
            (normrnd(0, 1, N, n_ofdm_symbols) + i * normrnd(0, 1, N, n_ofdm_symbols));
    
    enhanced_noise = (1./lambda) .* (W*noise);
        
    rx_ofdm_symbols = tx_ofdm_symbols + enhanced_noise;
    
    %%%---QUANTIZATION---%%%

    rx_ofdm_qsymbols = zeros(size(rx_ofdm_symbols));
    
    %FIX IF IT WORKS FOR STUFF
    for k=1:oversampling,
        noise = (1 / sqrt(10^(snrdb(m)/10))) * (1/sqrt(2)) * ...
                (normrnd(0, 1, N, n_ofdm_symbols) + ...
                i * normrnd(0, 1, N, n_ofdm_symbols));
            
        enhanced_noise = (1./lambda) .* (W*noise);    
        
        temp = tx_ofdm_symbols + enhanced_noise;
        rx_ofdm_qsymbols = rx_ofdm_qsymbols + sign(temp);
    end

    %%%---DE-OFDM---%%%
    
    %rx_symbols = reshape(inv(Heig)*W*rx_ofdm_symbols, N*n_ofdm_symbols, 1);
    %rx_qsymbols = reshape(inv(Heig)*W*rx_ofdm_qsymbols, N*n_ofdm_symbols, 1);
    
    rx_symbols = rx_ofdm_symbols;
    rx_qsymbols = rx_ofdm_qsymbols;

    %%%---DEMODULATE DATA---%%%

    %--- HARD DECISION ---%
    rx_samples_hard = qamdemod(rx_symbols, 4);
    rx_cbits_hard = de2bi(rx_samples_hard)';
    rx_cbits_hard = reshape(rx_cbits_hard, block_size, num_blocks);
    %convert hard decisions to LLR
    rx_cbits_hard_llr = - (rx_cbits_hard * 2 - 1) * Inf;
    
    %for k = 1:num_blocks,
    %    rx_bits_hard(:, k) = step(hDec, rx_cbits_hard_llr(:,k));
    %end
    %
    %ber_hard(m) = sum(sum(abs(rx_bits_hard - bits))) / n_bits;
    %bler_hard(m) = sum(sign(sum(abs(rx_bits_hard - bits)))) / num_blocks;

    ber_uncoded(m) = sum(sum(abs(rx_cbits_hard - cbits))) / (block_size * num_blocks);
    
    %--- SOFT DECISION ---%
    %compute LLR = log ( P(b = 0) / P(b = 1) )
    sigma = (1 ./ (abs(lambda).*sqrt(10^(snrdb(m)/10)))); %per channel noise level
    rx_cbits_b1 = ((imag(rx_symbols) + 1/sqrt(2)).^2 - (imag(rx_symbols) - 1/sqrt(2)).^2) ./ (2*sigma.^2);
    rx_cbits_b2 = ((real(rx_symbols) - 1/sqrt(2)).^2 - (real(rx_symbols) + 1/sqrt(2)).^2) ./ (2*sigma.^2);
    rx_cbits_b1 = reshape(rx_cbits_b1, N*n_ofdm_symbols, 1);
    rx_cbits_b2 = reshape(rx_cbits_b2, N*n_ofdm_symbols, 1);
    rx_cbits_soft = reshape([rx_cbits_b1 rx_cbits_b2]', block_size, num_blocks);

    for k = 1:num_blocks,
        rx_bits(:, k) = step(hDec, rx_cbits_soft(:,k));
    end

    ber_soft(m) = sum(sum(abs(rx_bits - bits))) / n_bits;
    bler_soft(m) = sum(sign(sum(abs(rx_bits - bits)))) / num_blocks;
    
    %--- QUANTIZED SOFT DECISION ---%
    sigma = (1 ./ (abs(lambda).*sqrt(10^(snrdb(m)/10)))); %per channel noise level
    rx_qcbits_b1 = ((imag(rx_qsymbols) + 1/sqrt(2)).^2 - (imag(rx_qsymbols) - 1/sqrt(2)).^2) ./ (2*sigma.^2);
    rx_qcbits_b2 = ((real(rx_qsymbols) - 1/sqrt(2)).^2 - (real(rx_qsymbols) + 1/sqrt(2)).^2) ./ (2*sigma.^2);
    rx_qcbits_b1 = reshape(rx_qcbits_b1, N*n_ofdm_symbols, 1);
    rx_qcbits_b2 = reshape(rx_qcbits_b2, N*n_ofdm_symbols, 1);
    rx_qcbits_soft = reshape([rx_qcbits_b1 rx_qcbits_b2]', block_size, num_blocks);
    
    for k = 1:num_blocks,
        rx_qbits(:, k) = step(hDec, rx_qcbits_soft(:,k));
    end

    ber_quantized(m) = sum(sum(abs(rx_qbits - bits))) / n_bits;
    bler_quantized(m) = sum(sign(sum(abs(rx_qbits - bits)))) / num_blocks;
end

%%%--- DATA SAVING ---%%%
timestamp = datestr(now, 'yyyymmdd-HHMM');
filename = ['data/', timestamp, '_cp=', num2str(cp)];
save(filename, ...
    'num_blocks', 'snrdb', 'oversampling', ...
    'bits', 'cbits', ...
    'ber_uncoded', 'ber_soft', 'bler_soft', ...
    'ber_quantized', 'bler_quantized', 'cp')