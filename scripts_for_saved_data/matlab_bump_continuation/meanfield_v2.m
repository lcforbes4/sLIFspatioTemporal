function v0 = meanfield_v2(J0,J1,E, N, dt, totaltime, equil_choice, pertstrength, spatialdependence, plotevery, v_init)
    % runs an euler time sim for the mean field and returns v at the final
    % time step

L = 2*pi;
dx = L/N;
%x = [0:dx:L-dx]';%linspace(0, L, N)';
%x(2) - x(1)
x   = linspace(0, L,N+1)'; x=x(1:end-1);
    %derivative vectors (in Fourier space)
k   = ([[0:N/2] [-N/2+1: -1]])';

f = @(v) (v-1).*heaviside(v-1);

vplus = (J0+sqrt(J0^2+4*(E-J0)))/2;%%only the plus branch
vminus = (J0-sqrt(J0^2+4*(E-J0)))/2;

if nargin < 11 %if not given v_init
    if equil_choice == 0
        v0 = ones(N,1)*vplus;
    elseif equil_choice == 1
        v0 = ones(N,1)*vminus; 
    elseif equil_choice ==2 
        v0 = ones(N,1)*E; %silent state
    end
else
    v0=v_init;
end

if spatialdependence
    %v0 = v0 +pertstrength*randn(N,1)/N; %add some noise to the initial condition
    v0 = v0 + pertstrength*cos(x);
else
    v0 = v0+pertstrength*ones(N,1);
end

% h = figure(1)
% plot(x,v0)
% %axis([0 L 0 vstar+1])
% xlabel('$x$', 'interpreter', 'latex')
% ylabel('$v(x,t)$', 'interpreter', 'latex')
% %pause


J = (J0 + J1*cos(x))/2/pi;
Jhat = fft(J);
%
%figure(2)
%plot(fftshift(k),fftshift(abs(Jhat)/N))


v0hat = fft(v0);
t = 0;
E = E*ones(N,1);
stepcounter = 0;
plotstep = round(plotevery/dt);


while t < totaltime
%     v1hat = v0hat + dt*(fft(E)-v0hat-fft(f(v0).*(v0))+(2*pi)/N*Jhat.*fft(f(v0)));
%     v1 = real(ifft(v1hat));
    v1 = v0 + dt*(E - v0 - f(v0).*v0 + 2*pi/N*ifft(Jhat.*fft(f(v0)))); 

    % if mod(stepcounter, plotstep) == 0
    %     figure(1)
    %     %hold on
    %     plot(x, v1);
    %     title(['t = ' num2str(t)])
    %     xlabel('Space (x)')
    %     ylabel('Voltage (v)')
    %     %hold off
    %     %axis([0 L 0 vstar+1])
    %     drawnow;
    % end
    
    v0 = v1;
    %v0hat = fft(v0);
    t = t + dt;
    stepcounter = stepcounter + 1;
end

end