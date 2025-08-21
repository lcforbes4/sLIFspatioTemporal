%%%continuation script - using spectral Newton-GMRS
clear all 
close all
clear; clc;

%% continuation parameters
ds = 0.005;
maxstep = 500;
arclength_max = 0.01;

%set domain and initial system parameters
L = 2*pi;
N = 2^12; 
dx = L/N;
dt = 0.01; %only for time steppers;
J1 = 10;
J0 = -1.2;
E0 = 1.000001; 
par_direc = [-1 1];
check = 1; %0 for v+, 1 for v-, and 2 for E
%
x   = linspace(0, L,N+1)'; 
x=x(1:end-1);
norm_tol = 0.001;

    %derivative vectors (in Fourier space)
k   = ([[0:N/2] [-N/2+1: -1]])';

%newton parameters
tol = 1e-8;
niter = 100;
maxiter = 20;
nminstep = 1e-14;
nitermax =10;
newtonflag=0;

%% set up figure:
h=figure(4);


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
    pertstrength = 50;%1e-1;
    spatialdependence = 1; % 0 for spatially homogeneous perturbation, 1 for spatial white noise ie perturb*cos

% run time stepper to get good initial guess
    v0 = meanfield_v2(J0,J1,E0,N, dt, totaltime, check, pertstrength, spatialdependence, plotevery);



%% %%%%%%%%Initial continuation %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u=[v0;0]; % append parameter line
F0 =@(uu) lif_u(uu,Jhat, E0,f,k,sx,N); %Initial root find?

% Set linear solver tolerance to be about 10 times smaller than the Newton
gmrestol=tol*1e-2;
gmaxit = 100;
gminner = 10;
rhs = F0(u);
figure(2)
plot(rhs)
title('F0([v0;0]) RHS')
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

        figure(5)
        hold on
        plot(x,real(u(1:end-1)))
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

figure(3)
plot(x,u(1:end-1))
title('result of convergence at initial parameters')
c0 = u(end);


%% %%%%%%Now start secant continuation in parameter E

%%%%%%%%%%%
J0_grid = zeros(maxstep,1); J0_grid(1) = J0; 
vstar = (J0+sqrt(J0^2+4*(E0-J0)))/2;
cgrid = J0_grid; cgrid(1) = c0;
l2grid = J0_grid; l2grid(1) = sqrt(dx)*norm(u(1:end-2)-vstar,2);
SOL_grid = [u;J0];

mingrid = J0_grid; mingrid(1) = min(u(1:end-1));
maxgrid = J0_grid; maxgrid(1) = max(u(1:end-1));



steps = 1;
for pp = par_direc

    %initialize secant, stepping in par_direc
    sec = [zeros(N+1,1);pp];
    zold = [u;J0]; % The initial solution
    z0 = zold+0.01*sec; % step it in one direction of the parameter
    zinit = z0; % the stepped version as initial condition of continuation
    ii=1;

    %kernel
    J = (z0(end) + J1*cos(x))/ 2/pi;
    Jhat = fft(J);

    %%%%%BEGIN continuation %%%%%
    maxJ0=10;
    while (steps < maxstep) ...
            && (sqrt(dx) * norm(zold(1:end-2)- ((J0+sqrt(J0^2+4*(zold(end)-J0)))/2),2)  >norm_tol)...
            && (sqrt(dx) * norm(zold(1:end-2)- ((J0-sqrt(J0^2+4*(zold(end)-J0)))/2),2)  >norm_tol) ...
            && (sqrt(dx) * norm(zold(1:end-2)- zold(end),2)                             >norm_tol)...
            && z0(end)<maxJ0 %to stop it from going off to pos infinity
        ii=ii+1;

        % compute initial residual
        [nrhs] =lif_z_J0(z0,zinit,sec,E0,Jhat,f,k,sx,N);
        nresidual = norm(nrhs);

        %%%%%%%%%%%%%%%%%%%%%%%Newton loop%%%%%%%%%%%%%%
        niter = 1; %counter for Newton
        tic
        fprintf(['starting Newton with J0 = ' num2str(zinit(end))])
        while (nresidual>tol)
            % define jacobian:
            dgt = @(dz) dlif_z_J0(dz,z0,zinit,sec,Jhat,f,f1,k,sx,N);

            fprintf([' solving linear system with residual ' num2str(nresidual) '...']);
            [nincr,flag,relres,iter,resvec] = gmres(dgt,nrhs,gminner,gmrestol,gmaxit);

            % error if linear solver didn't converge:
            if flag >0  sprintf(['gmres did not converge in initial problem, residual is ' num2str(residual) ' after ' num2str(niter) ' iterations'])
                newtonflag=2;
                break
            end

            % Newton step:
            z0 = z0-nincr;
            z0 = real(z0);%seemingly have to do this...

            % recompute residual and estimate norm:
            J = (z0(end) + J1*cos(x))/ 2/pi;
            Jhat = fft(J);

            [nrhs] = lif_z_J0(z0,zinit,sec,E0,Jhat,f,k,sx,N);
            nresidual = norm(nrhs);
            disp(['nres=' num2str(nresidual)  ', last gmresiters outer=' num2str(iter(1)) ', inner=' num2str(iter(2))])

            % errors:
            if ((niter>nitermax)&&steps>1 )|| (niter>30)
                disp(['Maximal number of iterations reached in large domain problem, giving up; residual is ' num2str(nresidual)])
                newtonflag=1;
                break
            end
            %
            if  norm(nincr)<nminstep
                disp(['Newton step is ineffective, giving up; residual is ' num2str(nresidual)])
                newtonflag=1;
                break
            end
            %
            if  (nresidual>100)&&(steps>1)
                disp(['Residual too large, is ' num2str(nresidual)])
                newtonflag=1;
                break
            end
        end
        disp(['#Newton iters=' num2str(niter-1)  ', J0=' num2str(z0(end)) ', c=' num2str(z0(end-1)) ', time=' num2str(toc)]) % output summary from this step
        %%%%%%%%%%%%%%%%%%%%%END Newton Loop%%%%%%%%%%%%%

        newton_control='none'; % perform tests on necessary grid changes
        if (newtonflag==1) || (newtonflag ==2)
            newton_control='reduce_step';
        end

        % Adjust newton loop based on performance:
        switch newton_control
            case 'reduce_step'
                ds = ds/2;
                disp(['Reduce step size to  ' num2str(ds)])
                newtonflag=0;
                zinit=zold+(steps>2)*ds*sec+(steps==2)*1e-4*ds*sec; % only march after step 2
                z0=zinit;
                %kernel
                J = (z0(end) + J1*cos(x))/ 2/pi;
                Jhat = fft(J);
            otherwise %if everything went fine
                steps = steps+1;JJ = z0(end);
                J0_grid(steps) = JJ; vstar = (JJ+sqrt(JJ^2+4*(E0-JJ)))/2;
                cgrid(steps) = z0(end-1);
                l2grid(steps) = sqrt(dx)*norm(z0(1:end-2)-vstar,2);
                
                mingrid(steps) = min(z0(1:end-2));
                maxgrid(steps) = max(z0(1:end-2));

                %SOL_grid(:,steps) = z0;
                SOL_grid = [SOL_grid;z0];


                set(0,'CurrentFigure',h);
                subplot(3,1,1)
                plot(x,z0(1:end-2)-vstar,'.-')
                xlabel('$x$','Interpreter','latex')
                ylabel('$v-v_*$','Interpreter','latex')

                subplot(4,1,2)
                plot(x,z0(1:end-2),'.-')
                xlabel('$x$','Interpreter','latex')
                ylabel('$v$','Interpreter','latex')

                subplot(4,1,3)
                plot(J0_grid(1:steps),l2grid(1:steps),'.-')
                xlabel('$J0$','Interpreter','latex')
                ylabel('$\| v - v_*\|_{L^2}$','Interpreter','latex')

                subplot(4,1,4)
                plot(J0_grid(1:steps),mingrid(1:steps),'b.-')
                hold on
                plot(J0_grid(1:steps),maxgrid(1:steps),'r.-')
                xlabel('$J0$','Interpreter','latex')
                ylabel('min/max$(v)$','Interpreter','latex')

                drawnow


                if steps>1
                    sec=z0-zold;
                    sec = sec/norm(sec);%update secant
                end

                ds = min(ds*1.4,arclength_max);

                % march one step in the direction of the previous secant
                zold=z0;
                zinit=z0+ds*sec; % only march after step 1; in step 2 reduce marching since only in direction of parameter
                z0=zinit;

                %kernel
                J = (z0(end) + J1*cos(x))/ 2/pi;
                Jhat = fft(J);
        end % end of newton_control



    end %%%%END CONTINUATION
    %Flip around array after 1st direction
    if pp == par_direc(1)
        J0_grid(1:steps) = flipud(J0_grid(1:steps));
        cgrid(1:steps) = flipud(cgrid(1:steps));
        l2grid(1:steps) = flipud(l2grid(1:steps));
        SOL_grid(1:steps) = flipud(SOL_grid(1:steps));
        mingrid(1:steps) = flipud(mingrid(1:steps));
        maxgrid(1:steps) = flipud(maxgrid(1:steps));
    end

end
E = E0;
save('lif_data_J0_E_-1.5.mat','J0_grid','cgrid','l2grid','SOL_grid','mingrid','maxgrid','N','x','L','E','J1')

figure(13)
plot(J0_grid(1:steps),mingrid(1:steps),'b.-')
hold on
plot(J0_grid(1:steps),maxgrid(1:steps),'r.-')
plot(J0_grid(1:steps),-.9,'g.-')
plot(J0_grid(1:steps),( J0_grid(1:steps)+ sqrt(J0_grid(1:steps).^2 + 4*(-.9 - J0_grid(1:steps)) ))/2,'y.-')
plot(J0_grid(1:steps),(J0_grid(1:steps)-sqrt(J0_grid(1:steps).^2 + 4*(-.9 - J0_grid(1:steps))))/2,'k.-')