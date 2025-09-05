%%%jacobian function for initial root find (no continuation equation
function [DF] = dlif_z(dz,z,zinit,sec,Jhat,f,f1,k,sx,N)
dv = dz(1:N);
dc = dz(N+1);
dE = dz(N+2); %continuation parameter not necs E

v=z(1:N); 
c=z(N+1);
%E = z(N+2);


DF1 = (-1 - f(v) - v.*f1(v)).*dv...
    + 2*pi/N*ifft(Jhat.*fft(f1(v).*dv))+ifft(1i*c*k.*fft(dv))...
    + ifft(1i*k.*fft(v))*dc...
    + ones(N,1)*dE;%deriv of DE wrt E
DF2 = sx'*dv;
DF3 = sum(dz.*sec);

DF = [DF1;DF2;DF3];




end