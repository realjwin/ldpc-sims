clear all; close all;

%variables
N = 64;
num_blocks = N; %this is number of codeblocks sent, must be multiple of N
oversampling = 1;
snrdb = [0, 1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1];
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
w = exp(-2*pi*i/N);
W = zeros(N,N);
for j=0:N-1,
    for k=0:N-1,
        W(j+1,k+1) = w^(j*k)/sqrt(N);
    end
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
    
    %%%---MODULATE DATA---%%%
    cbits_temp = reshape(cbits, 2, n_cbits/2)';
    %bi2de is little-endian, largest bit = highest index
    %assume everything is little-endian!
    tx_samples = bi2de(cbits_temp);

    %qpsk modulation
    %from quadrant 1 to 4 (couterclockwise): 10, 00, 01, 11
    %note: I'm treating this as little-endian
    tx_symbols = (1/sqrt(2)) * qammod(tx_samples, 4);

    %%%---OFDM---%%%

    %each ofdm symbol is a column of length N
    tx_ofdm_symbols = W'*reshape(tx_symbols, N, n_ofdm_symbols);

    %%%---CHANNEL---%%%

    %channel is Rayleigh(1)
    %note: CP not needed, incorporating in H
    cp = 1;
    h = normrnd(0,n_ofdm_symbols,1,cp) + i*normrnd(0,n_ofdm_symbols,1,cp);
    %normalize h
    h = h ./ sqrt(sum(abs(h).^2, 2));

    %flip h to retain expectation that h(1) is the first elt
    %generate proper circular channel via bullshit
    temp = [fliplr(h) zeros(n_ofdm_symbols,N-cp)];
    H = toeplitz([temp(1) fliplr(temp(2:end))], temp);
    H = circshift(H, -(cp-1), 2);
    %H = eye(N)

    %generate channel orthogonalization
    Heig = W*H*W';

    %noise
    noise = (1 / sqrt(10^(snrdb(m)/10))) * (1/sqrt(2)) * ...
            (normrnd(0,1,N,n_ofdm_symbols) + i * normrnd(0,1,N,n_ofdm_symbols));
    rx_ofdm_symbols = H*tx_ofdm_symbols + noise;

    %%%---QUANTIZATION---%%%

    rx_ofdm_qsymbols = zeros(size(rx_ofdm_symbols));
    
    for k=1:oversampling,
        noise = (1 / sqrt(10^(snrdb(m)/10))) * (1/sqrt(2)) * ...
        (normrnd(0,1,N,n_ofdm_symbols) + i * normrnd(0,1,N,n_ofdm_symbols));
        temp = H*tx_ofdm_symbols + noise;
        rx_ofdm_qsymbols = rx_ofdm_qsymbols + sign(temp);
    end

    %%%---DE-OFDM---%%%

    rx_symbols = reshape(inv(Heig)*W*rx_ofdm_symbols, N*n_ofdm_symbols, 1);
    rx_qsymbols = reshape(inv(Heig)*W*rx_ofdm_qsymbols, N*n_ofdm_symbols, 1);

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
    sigma = (1 / sqrt(10^(snrdb(m)/10)));  %CHANGE per channel
    rx_cbits_b1 = ((imag(rx_symbols) + 1/sqrt(2)).^2 - (imag(rx_symbols) - 1/sqrt(2)).^2) / (2*sigma^2);
    rx_cbits_b2 = ((real(rx_symbols) - 1/sqrt(2)).^2 - (real(rx_symbols) + 1/sqrt(2)).^2) / (2*sigma^2);
    rx_cbits_soft = reshape([rx_cbits_b1 rx_cbits_b2]', block_size, num_blocks);

    for k = 1:num_blocks,
        rx_bits(:, k) = step(hDec, rx_cbits_soft(:,k));
    end

    ber_soft(m) = sum(sum(abs(rx_bits - bits))) / n_bits;
    bler_soft(m) = sum(sign(sum(abs(rx_bits - bits)))) / num_blocks;
    
    %--- QUANTIZED SOFT DECISION ---%
    sigma = (1 / sqrt(10^(snrdb(m)/10))); %CHANGE per channel
    rx_qcbits_b1 = ((imag(rx_qsymbols) + 1/sqrt(2)).^2 - (imag(rx_qsymbols) - 1/sqrt(2)).^2) / (2*sigma^2);
    rx_qcbits_b2 = ((real(rx_qsymbols) - 1/sqrt(2)).^2 - (real(rx_qsymbols) + 1/sqrt(2)).^2) / (2*sigma^2);
    rx_qcbits_soft = reshape([rx_qcbits_b1 rx_qcbits_b2]', block_size, num_blocks);
    
    for k = 1:num_blocks,
        rx_qbits(:, k) = step(hDec, rx_qcbits_soft(:,k));
    end

    ber_quantized(m) = sum(sum(abs(rx_qbits - bits))) / n_bits;
    bler_quantized(m) = sum(sign(sum(abs(rx_qbits - bits)))) / num_blocks;
end

%%%--- DATA SAVING ---%%%
timestamp = datestr(now, 'yyyymmdd-HHMM');
filename = ['data/', timestamp, '_ldpcsim'];
save(filename, ...
    'num_blocks', 'snrdb', 'oversampling', ...
    'bits', 'cbits', ...
    'ber_uncoded', 'ber_soft', 'bler_soft', ...
    'ber_quantized', 'bler_quantized')