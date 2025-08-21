%%%function for initial root find (no continuation equation
function [F] = lif_u(u,Jhat, E,f,k,sx,N)

v=u(1:end-1); 
c=u(end);

F1 = E - v - f(v).*v + 2*pi/N*ifft(Jhat.*fft(f(v))+1i*c*k.*v); 
F2 = sx'*v; %phase condition (simple)

F = [F1;F2];



end