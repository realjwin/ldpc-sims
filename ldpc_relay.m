clear all; close all;

%---VARIABLES---%
load('parity.mat') %(324, 648) 802.11n HT LDPC
hEnc = comm.LDPCEncoder('ParityCheckMatrix', sparse(H));
iterations = [2 4 8 16 32];
num_blocks = 1000;
snrdb = [0:.5:5];

%---COMPUTATION---%
for n=1:num_blocks,
    disp(['num_blocks = ' num2str(n)])
    
    %encode
    bits = logical(randi([0 1], block_size*rate, 1));
    cbits = hEnc(bits);
    
    for m=1:length(snrdb),
        %channel (awgn)
        sigma = 1/sqrt(10^(snrdb(m)/10));
        noise = normrnd(0, sigma, block_size, 1);

        %channel (bpsk)
        mcbits = 2 .* cbits - 1;
        rmcbits = mcbits + noise;
        rscbits = ((rmcbits - 1).^2 - (rmcbits + 1).^2) / (2*sigma^2);

        %decoding
        for k=1:length(iterations),
            %disp(['max iter #: ', num2str(max_iter(k))])
            
            %init decoder
            hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H), ...
            'DecisionMethod', 'Soft decision', ...
            'OutputValue', 'Whole codeword', ...
            'MaximumIterationCount', iterations(k));

            rbits_final{k,m}(:, n) = hDec(rscbits);
        end
    end
    
    cbits_final(:, n) = cbits;
end

%---BER---%
for m=1:length(snrdb),
    for k=1:length(iterations),
        temp = cbits_final - .5 * ( - sign(rbits_final{k,m}) + 1 );
        ber_final(k, m) = sum(sum(abs( temp(1:block_size*rate, :) ))) ...
                            / (block_size*rate*num_blocks);
    end
end

%---SAVE DATA---%

%CBITS: each column is a codeword, there are num_blocks codewords

%RBITS_FINAL: {k,m} specifies iterations(k) number of iterations at snrdb(m)
%then inside each of those is the same as cbits.

%BER_FINAL: (k,m) specifies BER with iterations(k) # of iterations at
%snrdb(m)

timestamp = datestr(now, 'yyyymmdd-HHMM');
filename = ['data/ml/' timestamp '_relay_n=' num2str(num_blocks)];

save(filename, 'iterations', 'snrdb', 'cbits_final', 'rbits_final', ...
                'ber_final', 'num_blocks'); 

%---PLOT DATA---%

for k=1:length(iterations),
    semilogy(snrdb, ber_final(k,:))
    hold on
end

%also compare performance when we ADD the two LLR's together & compare that
%with what happen when we use a deep learning based approach to the same
%thing

%what kind of things can I take away from the LDPC for relay codes paper?
%what kind of things can I take away from Hyeji Kim's paper?