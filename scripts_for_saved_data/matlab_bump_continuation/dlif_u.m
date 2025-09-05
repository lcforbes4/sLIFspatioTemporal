%%%jacobian function for initial root find (no continuation equation
function [DF] = dlif_u(du,u,Jhat, E,f,f1,k,sx,N)

dv = du(1:end-1);
dc = du(end);

v=u(1:end-1); 
c=u(end);


DF1 = (-1 - f(v) - v.*f1(v)).*dv...
    + 2*pi/N*ifft(Jhat.*fft(f1(v).*dv))+ifft(1i*c*k.*fft(dv))...
    + ifft(1i*k.*fft(v))*dc;
DF2 = sx'*dv;

DF = [DF1;DF2];




end