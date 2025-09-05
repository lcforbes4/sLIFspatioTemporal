%%%function for initial root find (no continuation equation
function [F] = lif_z(z,zinit,sec,Jhat,f,k,sx,N)
v=z(1:N); 
c=z(N+1);
E = z(N+2);

F1 = E - v - f(v).*v + 2*pi/N*ifft(Jhat.*fft(f(v))+1i*c*k.*v); 
F2 = sx'*v; %phase condition (simple)
F3 = sum((z-zinit).*sec); %continuation orthogonality condition

F = [F1;F2;F3];



end