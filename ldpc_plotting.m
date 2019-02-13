%%%--- PLOTTING ---%%% (make this different file)
ber_theory = qfunc(sqrt(10.^(snrdb./10)));

figure(1)
semilogy(snrdb, ber_soft)
hold on
semilogy(snrdb, ber_quantized)
semilogy(snrdb, ber_uncoded, '*')
semilogy(snrdb, ber_theory)
title('BER')
legend('Soft', 'Quantized', 'Uncoded', 'Theory')

%figure(2)
%semilogy(snrdb, bler_soft)
%hold on
%semilogy(snrdb, bler_hard)
%title('BLER')
%legend('Soft','Hard')