clear all; close all;

%variables
N = 64;
cp = 12; %specifies # of channel taps (taps = cp+1)
num_blocks = 2*N; %5*N; %this is number of codeblocks sent, must be multiple of N
oversampling = 1;
coherence = 1; %120ms, 12 ofdm sym/ms
snrdb = [0, 1, 1.3, 1.6:.2:7];
n_snrdb = length(snrdb);
plot = 1;

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
    
    %%%---VARIABLE RESET---%%%
    lambda = zeros(N*n_ofdm_symbols,1);
    rx_symbols = zeros(N, n_ofdm_symbols);
    rx_qsymbols = zeros(N, n_ofdm_symbols);

    
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
    
    for k=1:n_ofdm_symbols/coherence,
        %channel is Rayleigh(1)
        h = normrnd(0,1,1,cp+1) + i*normrnd(0,1,1,cp+1);
        %normalize h
        h = h ./ sqrt(sum(abs(h).^2, 2));

        %flip h to retain expectation that h(1) is the first elt
        %generate proper circular channel via bullshit
        temp = [fliplr(h) zeros(1,N-(cp+1))];
        H = toeplitz([temp(1) fliplr(temp(2:end))], temp);
        H = circshift(H, -cp, 2);
        %H = eye(N)

        %generate channel orthogonalization (eigenvalues)
        Heig{k} = W*H*W';
        
        lambda((k-1)*N*coherence+1:k*N*coherence) = ...
            repmat(diag(Heig{k})', 1, coherence);
                
        rx_ofdm_symbols(:, (k-1)*coherence+1:k*coherence) = ...
            H*W'*tx_ofdm_symbols(:, (k-1)*coherence+1:k*coherence);
    end

    %%%---QUANTIZATION & NOISE---%%%

    %generate noise enhancement matrix [ inv(L)*W ]
    %y = HW'x+n, inv(L)*Wy = x + inv(L)*Wn
    
    rx_ofdm_qsymbols = zeros(size(rx_ofdm_symbols));
    
    for k=1:oversampling,
        noise = (1 / sqrt(10^(snrdb(m)/10))) * (1/sqrt(2)) * ...
                (normrnd(0, 1, N, n_ofdm_symbols) + ...
                i * normrnd(0, 1, N, n_ofdm_symbols));
        
        temp = rx_ofdm_symbols + noise;
        rx_ofdm_qsymbols = rx_ofdm_qsymbols + sign(temp);
    end

    %add noise for non-quantized
    noise = (1 / sqrt(10^(snrdb(m)/10))) * (1/sqrt(2)) * ...
            (normrnd(0, 1, N, n_ofdm_symbols) + ...
            i * normrnd(0, 1, N, n_ofdm_symbols));
    
    rx_ofdm_symbols = rx_ofdm_symbols + noise;

    %%%---DE-OFDM---%%%
    
    for k=1:n_ofdm_symbols/coherence,
        rx_symbols(:, (k-1)*coherence+1:k*coherence) = ...
            inv(Heig{k})*W*rx_ofdm_symbols(:, (k-1)*coherence+1:k*coherence);
        rx_qsymbols(:, (k-1)*coherence+1:k*coherence) = ...
            inv(Heig{k})*W*rx_ofdm_qsymbols(:, (k-1)*coherence+1:k*coherence);
    end
    
    %reshape(inv(Heig)*W*rx_ofdm_symbols, N*n_ofdm_symbols, 1);
    
    %%%---DEMODULATE DATA---%%%
    
    rx_symbols = reshape(rx_symbols, N*n_ofdm_symbols, 1);
    rx_qsymbols = reshape(rx_qsymbols, N*n_ofdm_symbols, 1);

    %--- HARD DECISION ---%
    rx_samples_hard = qamdemod(rx_symbols, 4);
    rx_cbits_hard = de2bi(rx_samples_hard)';
    rx_cbits_hard = reshape(rx_cbits_hard, block_size, num_blocks);
    %convert hard decisions to LLR
    %rx_cbits_hard_llr = - (rx_cbits_hard * 2 - 1) * Inf;

    ber_uncoded(m) = sum(sum(abs(rx_cbits_hard - cbits))) / (block_size * num_blocks);
    
    rx_qsamples_hard = qamdemod(rx_qsymbols, 4);
    rx_qcbits_hard = de2bi(rx_qsamples_hard)';
    rx_qcbits_hard = reshape(rx_qcbits_hard, block_size, num_blocks);
    %convert hard decisions to LLR
    %rx_qcbits_hard_llr = - (rx_qcbits_hard * 2 - 1) * Inf;
    
    ber_quncoded(m) = sum(sum(abs(rx_qcbits_hard - cbits))) / (block_size * num_blocks);
    
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
    'ber_quantized', 'bler_quantized', 'cp', 'coherence')

if plot == 1,
    ber_theory = qfunc(sqrt(10.^(snrdb./10)));
    
    semilogy(snrdb, ber_theory, 'k')
    hold on
    semilogy(snrdb, ber_uncoded, '*k')
    semilogy(snrdb, ber_quncoded, '*b')
    semilogy(snrdb, ber_soft, '-rs')
    semilogy(snrdb, ber_quantized, '-bo')
    xlabel('SNR (dB)')
    ylabel('BER')
    legend('Theory', 'Uncoded', 'Quantized Uncoded', 'Soft', 'Quantized') %add quantized uncoded
end