function Xn=AddNoise(X,SNR)
Xn=zeros(size(X));

for i=1:size(X,2)
    Noise=randn(size(X,1),1);
    Xn(:,i)=X(:,i)+(norm(X(:,i))/(10^(SNR/20)))*Noise/(norm(Noise));
end
end