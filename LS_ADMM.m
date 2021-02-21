function O=LS_ADMM(T, Y, eps_o, mu, kmax)
p=size(Y,1);
q=size(T,1);
Lam=zeros(q,p);
YYT=Y*Y';
temp=inv( YYT + (1/mu) * eye(p) );
TYT=T*Y';
Z=Lam;

MyTemp=[];
for iter=1:kmax
    % O-update
    O=(TYT+(1/mu)*(Z+Lam))*temp;
    % Z-update
    Z=O-Lam;
    nz=norm(Z,'fro');
    if nz > eps_o
        Z=Z*(eps_o/nz);
    end
    % Lam-update
    Lam=Lam+Z-O;
    
%     MyTemp=[MyTemp,norm(Lam,'fro')];
%     figure(50)
%     plot(MyTemp)
%     hold on
%     drawnow

end
