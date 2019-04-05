%%%--- PLOTTING ---%%% (make this different file)
load('data/20190325-1141_cp=3.mat')
ber_soft3 = ber_soft;
ber_quantized3 = ber_quantized;
ber_uncoded3 = ber_uncoded;
snrdb3 = snrdb;

load('data/20190325-1351_cp=2.mat')
ber_soft2 = ber_soft;
ber_quantized2 = ber_quantized;
ber_uncoded2 = ber_uncoded;
snrdb2 = snrdb;

load('data/20190325-2104_cp=1.mat')
ber_soft1 = ber_soft;
ber_quantized1 = ber_quantized;
ber_uncoded1 = ber_uncoded;
snrdb1 = snrdb;

load('data/20190326-0005_cp=0.mat')
ber_soft0 = ber_soft;
ber_quantized0 = ber_quantized;
ber_uncoded0 = ber_uncoded;
snrdb0 = snrdb;

ber_theory = qfunc(sqrt(10.^(snrdb0./10)));

figure(1)
semilogy(snrdb0, ber_uncoded, '*k')
hold on
semilogy(snrdb0, ber_theory, 'k')

semilogy(snrdb0, ber_soft0, '-r+')
semilogy(snrdb0, ber_soft1, '-rs')
semilogy(snrdb0, ber_soft2, '-ro')
semilogy(snrdb0, ber_soft3, '-r^')

semilogy(snrdb0, ber_quantized0, '-b+')
semilogy(snrdb0, ber_quantized1, '-bs')
semilogy(snrdb0, ber_quantized2, '-bo')
semilogy(snrdb0, ber_quantized3, '-b^')



title('BER')
legend('Uncoded', 'Theory', ...
    'Soft CP=0', 'Soft CP=1', 'Soft CP=2', 'Soft CP=3', ...
    'Quantized CP=0', 'Quantized CP=1', 'Quantized CP=2', 'Quantized CP=3')
xlabel('SNR (dB)')
ylabel('BER')

%figure(2)
%semilogy(snrdb, bler_soft)
%hold on
%semilogy(snrdb, bler_hard)
%title('BLER')
%legend('Soft','Hard')