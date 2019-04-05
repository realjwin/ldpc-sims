load('data/ml/20190326-2014_relay_n=1000.mat')

for k=1:length(iterations),
    semilogy(snrdb, ber_final(k,:))
    hold on
end

xlabel('SNR (dB)')
ylabel('BER')
legend('Iter: 2', '4', '8', '16', '32')