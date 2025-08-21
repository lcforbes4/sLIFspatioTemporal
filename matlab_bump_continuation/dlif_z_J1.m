%%%jacobian function for initial root find (no continuation equation
function [DF] = dlif_z(dz,z,zinit,sec,Jhat,f,f1,k,fftcx,sx,N)
dv = dz(1:N);
dc = dz(N+1);
dparam = dz(N+2); %continuation parameter

v=z(1:N); 
c=z(N+1);
%E = z(N+2);

x = linspace(0, 2*pi,N+1)'; 
x = x(1:end-1);

%vhat=fft(v)

DF1 = (-1 - f(v) - v.*f1(v)).*dv...
    + 2*pi/N*ifft(Jhat.*fft(f1(v).*dv))+ifft(1i*c*k.*fft(dv))...
    + ifft(1i*k.*fft(v))*dc...
    + 2*pi/N*ifft(fftcx.*fft(f(v)))*dparam;%deriv of DE wrt J1
DF2 = sx'*dv;
DF3 = sum(dz.*sec);

DF = [DF1;DF2;DF3];




end