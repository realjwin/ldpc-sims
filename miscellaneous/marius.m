% IEEE 802.11n HT LDPC
Z = 27;
rotmatrix = ...
   [0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
   22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
   6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
   2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
   23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
   24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
   25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
   13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
   7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
   11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
   25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
   3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];

H = zeros(size(rotmatrix)*Z);
Zh = diag(ones(1, Z), 0);

% Convert into binary matrix
for r=1:size(rotmatrix, 1)
   for c=1:size(rotmatrix, 2)
       rotidx = rotmatrix(r, c);
       if (rotidx > -1)
           Zt = circshift(Zh, [0 rotidx]);
       else
           Zt = zeros(Z);
       end
       limR = (r-1)*Z+1:r*Z;
       limC = (c-1)*Z+1:c*Z;
       H(limR, limC) = Zt;
   end
end

% Encoder/decoder objects
hEnc = comm.LDPCEncoder('ParityCheckMatrix', sparse(H));
hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H), 'DecisionMethod', 'Soft decision');

% System parameters
K = size(H, 1);
N = size(H, 2);

% Auxiliary tables for fast LLR computation
bitmap = de2bi(0:1).';
constellation = qammod(bitmap, 2, 'InputType', 'bit', 'UnitAveragePower', true);

% Random bits
bitsRef = randi([0 1], K, 1);
% Encode bits
bitsEnc = hEnc(bitsRef);
% Modulate bits with BPSK
x = qammod(bitsEnc, 2, 'InputType', 'bit', 'UnitAveragePower', true);
% Noise
noisePower = 1;
n = sqrt(noisePower) * randn(size(x));
y = x + n;

% Input LLR (closed-form)
llrIn = compute_llr(constellation, bitmap, 1, y, 1, noisePower);
% Output LLR - may need to change demodulation object to get all N output LLRs
llrOut = hDec(llrIn);

% Determine bit/packet error
bitsEst = (sign(-llrOut) +1) / 2;