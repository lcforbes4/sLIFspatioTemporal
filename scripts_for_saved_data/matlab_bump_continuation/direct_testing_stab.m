%direct_testing_stab.m
clc;
clear;
close all;

%%
%% continuation parameters
ds = 0.25;
maxstep = 1000;
arclength_max = 0.5;

%set domain and initial system parameters
L = 2*pi;
N = 2^12; 
dx = L/N;
dt = 0.01; %only for time steppers;
J1 = 10;
J0 = 4;
E0 = 0.01;%-1.444*J0 + 6.582;%-0.5; 
par_direc = [1,-1];
%check = 1; % IF WANT TO START AT V-

%
x = linspace(0, L,N+1)'; 
x = x(1:end-1);

%derivative vectors (in Fourier space)
k   = ([[0:N/2] [-N/2+1: -1]])';

%newton parameters
tol = 1e-6;
niter = 100;
maxiter = 20;
nminstep = 1e-14;
nitermax =10;
newtonflag=0;

%% set up figure:
% h=figure(4);


%kernel
J = (J0 + J1*cos(x))/ 2/pi;
sx = sin(x);

Jhat = fft(J);

%% define the firing rate function
f = @(v) (v-1).*heaviside(v-1);
f1 = @(v) heaviside(v-1);

%% Run Euler time-stepper

totaltime = 1;
plotevery = 5;

% Initial Condition Choices
    equil_choice = 0; % 0 for v+, 1 for v-, 2 for v_0=E
    pertstrength = .01;%1e-1;
    spatialdependence = 1; % 0 for spatially homogeneous perturbation, 1 for spatial perturbation ie perturb*cos

% run time stepper to get good initial guess
    %v0 = meanfield(J0,J1,E0,N, dt, totaltime, activity, pertstrength, spatialdependence, plotevery);
    v0 = meanfield_v2(J0,J1,E0,N, dt, totaltime, equil_choice, pertstrength, spatialdependence, plotevery);
    disp(v0)



%% %%%%%%%%Initial continuation %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u=[v0;0]; % append parameter line
F0 =@(uu) lif_u(uu,Jhat, E0,f,k,sx,N); %Initial root find?

% Set linear solver tolerance to be about 10 times smaller than the Newton
gmrestol=tol*1e-2;
gmaxit = 100;
gminner = 10;
rhs = F0(u);
% figure(2)
% plot(rhs)
% title('F0([v0;0]) RHS')
residual = norm(rhs);

% Newton loop using GMRES (with no preconditioner) to solve linear
% equation!!!
niter = 1;
while (residual>tol)&&(niter<maxiter)
    
    % form Jacobian:
        dft=@(du) dlif_u(du,u,Jhat,E0,f,f1,k,sx,N);
    % gmres solve for increment:
        [nincr,flag]=gmres(dft,rhs,gminner,gmrestol,gmaxit); 
    % Didn't Converge error:
        if flag==1
            sprintf(['gmres did not converge in initial problem, residual is ' num2str(residual) ' after ' num2str(niter) ' iterations'])
            ff='gmres';
            break
        end
    % Newton step:
        u=u-nincr; 

    % figure(5)
    % hold on
    % plot(x,real(u(1:end-1)))
    niter=niter+1;                % keep track of number of iterations
    
    % recompute residual
        u = real(u);
        rhs=F0(u);                  % compute residual
        residual = norm(rhs);          % estimate residual

    % Reached max iterations error:
    if niter>maxiter
        sprintf(['Maximal number of iterations reached in small domain problem, giving up; residual is ' num2str(npar.residual)])
        ff='newton';
        break
    end

end

figure(1)
hold on
plot(x,ones(N,1)*((J0-sqrt(J0^2+4*(E0-J0)))/2))
plot(x,ones(N,1)*((J0+sqrt(J0^2+4*(E0-J0)))/2))
plot(x,ones(N,1)*E0)
plot(x,u(1:end-1),'b--')
title('result of convergence at initial parameters')
c0 = u(end);

% time step the perturbation of converged solution
pertstrength1=.001;
spatialdependence1=0;
totaltime1=100;
v1 = meanfield_v2(J0,J1,E0,N, dt, totaltime1, equil_choice, pertstrength1, spatialdependence, plotevery, u(1:end-1)); %ignoring equil_choice since specifying IC

figure(2)
hold on
plot(x,ones(N,1)*((J0-sqrt(J0^2+4*(E0-J0)))/2))
plot(x,ones(N,1)*((J0+sqrt(J0^2+4*(E0-J0)))/2))
plot(x,ones(N,1)*E0)
plot(x,v1,'b--')
title('result of time stepping perturbed solution');