[x,Fs] = audioread('/a/a1.wav');
y = fft(x);
m = abs(y);
p = unwrap(angle(y));
f = (0:length(y)-1);

subplot(2,1,1)
plot(f,m)
title('Magnitude')

subplot(2,1,2)
plot(f,p)
title('Phase')

sgtitle('A Chord DFT');

achords = dir('a');
amchords = dir('am');
bmchords = dir('bm');
cchords = dir('c');
dchords = dir('d');
dmchords = dir('dm');
echords = dir('e');
emchords = dir('em');
fchords = dir('f');
gchords = dir('g');

output = zeros(10,10);

%loop through A chords, check against first entry of each chord class
for a=3:size(achords,1)
    [x,Fs] = audioread(achords(a).name); %pull a test chord
    x = x./norm(x); %normalize the wave
    results = zeros(10,1);
    
    %test against first chord of each class
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results); %find index of highest cross correlation energy
    output(1,i) = output(1,i) + 1; %add to confusion matrix
end

%repeat process for A chords for all chord classes
for a=3:size(amchords,1)
    [x,Fs] = audioread(amchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(2,i) = output(2,i) + 1;
end

for a=3:size(bmchords,1)
    [x,Fs] = audioread(bmchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(3,i) = output(3,i) + 1;
end

for a=3:size(cchords,1)
    [x,Fs] = audioread(cchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(4,i) = output(4,i) + 1;
end

for a=3:size(dchords,1)
    [x,Fs] = audioread(dchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(5,i) = output(5,i) + 1;
end

for a=3:size(dmchords,1)
    [x,Fs] = audioread(dmchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(6,i) = output(6,i) + 1;
end

for a=3:size(echords,1)
    [x,Fs] = audioread(echords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(7,i) = output(7,i) + 1;
end

for a=3:size(emchords,1)
    [x,Fs] = audioread(emchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(8,i) = output(8,i) + 1;
end

for a=3:size(fchords,1)
    [x,Fs] = audioread(fchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(9,i) = output(9,i) + 1;
end

for a=3:size(gchords,1)
    [x,Fs] = audioread(gchords(a).name);
    x = x./norm(x);
    results = zeros(10,1);
    
    [y,fs] = audioread(achords(3).name);
    y = y./norm(y);
    results(1) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(amchords(3).name);
    y = y./norm(y);
    results(2) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(bmchords(3).name);
    y = y./norm(y);
    results(3) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(cchords(3).name);
    y = y./norm(y);
    results(4) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dchords(3).name);
    y = y./norm(y);
    results(5) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(dmchords(3).name);
    y = y./norm(y);
    results(6) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(echords(3).name);
    y = y./norm(y);
    results(7) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(emchords(3).name);
    y = y./norm(y);
    results(8) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(fchords(3).name);
    y = y./norm(y);
    results(9) = norm(conv(x,y(end:-1:1)))^2;
    
    [y,fs] = audioread(gchords(3).name);
    y = y./norm(y);
    results(10) = norm(conv(x,y(end:-1:1)))^2;
    
    [m,i] = max(results);
    output(10,i) = output(10,i) + 1;
end

achords = dir('training_data_1/a');
amchords = dir('training_data_1/am');
bmchords = dir('training_data_1/bm');
cchords = dir('training_data_1/c');
dchords = dir('training_data_1/d');
dmchords = dir('training_data_1/dm');
echords = dir('training_data_1/e');
emchords = dir('training_data_1/em');
fchords = dir('training_data_1/f');
gchords = dir('training_data_1/g');
testchords = dir('test_data_1');

labels = zeros(10,1);

%loop through test chords, check against first entry of each chord class
for x = 3:12
    [y,Fs] = audioread(testchords(x).name); %pull a test chord
    y = y ./ norm(y); %normalize the wave
    res = zeros(10,1);
    
    %test the chord against first chord of each class
    [z,fs] = audioread(achords(3).name);
    z = z ./ norm(z);
    res(1) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(amchords(3).name);
    z = z ./ norm(z);
    res(2) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(bmchords(3).name);
    z = z ./ norm(z);
    res(3) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(cchords(3).name);
    z = z ./ norm(z);
    res(4) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(dchords(3).name);
    z = z ./ norm(z);
    res(5) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(dmchords(3).name);
    z = z ./ norm(z);
    res(6) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(echords(3).name);
    z = z ./ norm(z);
    res(7) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(emchords(3).name);
    z = z ./ norm(z);
    res(8) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(fchords(3).name);
    z = z ./ norm(z);
    res(9) = norm(conv(y,z(end:-1:1)))^2;
    
    [z,fs] = audioread(gchords(3).name);
    z = z ./ norm(z);
    res(10) = norm(conv(y,z(end:-1:1)))^2;
    
    [m,i] = max(res); %find index of highest cross correlation energy
    labels(x-2) = i; %store in labels
end
