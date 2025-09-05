%%%continuation script - using spectral Newton-GMRS
clear all 
close all
clear; clc;

%% continuation parameters
ds = 0.25;
maxstep = 7000;
arclength_max = 0.5;

%set domain and initial system parameters
L = 2*pi;
N = 2^12; 
dx = L/N;
dt = 0.01; %only for time steppers;
J1 = 10;
J0 = 3;
E0 = 1-.01;%-1.444*J0 + 6.582;%-0.5; 
par_direc = [1,-1];
check = 0; %0 for v_+

%
x = linspace(0, L,N+1)'; 
x = x(1:end-1);

%derivative vectors (in Fourier space)
k   = ([[0:N/2] [-N/2+1: -1]])';

%newton parameters
tol = 1e-4;
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
    activity = 1; % 0 for v0=E, 1 for v0=v+
    pertstrength = 1;%1e-1;
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
Egrid = zeros(maxstep,1); Egrid(1) = E0; 
vstar = (J0+sqrt(J0^2+4*(E0-J0)))/2;
cgrid = Egrid; cgrid(1) = c0;
l2grid = Egrid; l2grid(1) = sqrt(dx)*norm(u(1:end-2)-vstar,2);
SOL_grid = [u;E0];
eval_grid = zeros(maxstep,4);
eval1_grid = zeros(maxstep,1);
eval2_grid = zeros(maxstep,1);

mingrid = Egrid; mingrid(1) = min(u(1:end-1));
maxgrid = Egrid; maxgrid(1) = max(u(1:end-1));

steps = 1;
for pp = par_direc

    %initialize secant, stepping in par_direc
    sec = [zeros(N+1,1);pp];
    zold = [u;E0]; % The initial solution
    z0 = zold+0.01*sec; % step it in one direction of the parameter
    zinit = z0; % the stepped version as initial condition of continuation
    ii=1;

    %%%%%BEGIN continuation %%%%%
    norm_tol = 0.1;
    while (steps < maxstep) ...
        &&  (sqrt(dx)*norm(z0(1:end-2)-( (J0+sqrt(J0^2+4*(z0(end)-J0)))/2),2)>norm_tol) ...
        && (sqrt(dx)*norm(z0(1:end-2)-( (J0-sqrt(J0^2+4*(z0(end)-J0)))/2),2)>norm_tol) ...
        && (sqrt(dx)*norm(z0(1:end-2)-z0(end),2)>norm_tol)
        ii=ii+1;

        % compute initial residual
        [nrhs] =lif_z(z0,zinit,sec,Jhat,f,k,sx,N);
        nresidual = norm(nrhs);

        %%%%%%%%%%%%%%%%%%%%%%%Newton loop%%%%%%%%%%%%%%
        niter = 1; %counter for Newton
        tic
        fprintf(['starting Newton with E = ' num2str(zinit(end))])
        while (nresidual>tol)
            % define jacobian:
            dgt = @(dz) dlif_z(dz,z0,zinit,sec,Jhat,f,f1,k,sx,N);

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
            [nrhs] = lif_z(z0,zinit,sec,Jhat,f,k,sx,N);
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
        disp(['#Newton iters=' num2str(niter-1)  ', E=' num2str(z0(end)) ', c=' num2str(z0(end-1)) ', time=' num2str(toc)]) % output summary from this step
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
            otherwise %if everything went fine

                % Spectral Analysis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % remove continuation row and column ie just use dlif_u
                % instead of dlif_z
                jacobi = @(duu) dlif_uev(duu,z0(1:end-1), Jhat,z0(N+2), f,f1,k,sx,N); 

                [efcns,evals ]= eigs(jacobi, N,50,'largestreal','Tolerance',1e-15);
                evals = diag(evals);
                
                %figure(22);
                %hold on
                %plot(real(evals), imag(evals), '*');
                %title('Convergent eigenvalues')
                %xlim([-0.01,0.01])
                %ylim([-80,80])

                % plottn
                %figure(23);
                %plot(x,real(efcns(:,1:5)))
                %title('first 5 eigenfunctions')

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                steps = steps+1;EE = z0(end);
                Egrid(steps) = EE; vstar = (J0+sqrt(J0^2+4*(EE-J0)))/2;
                cgrid(steps) = z0(end-1);
                l2grid(steps) = sqrt(dx)*norm(z0(1:end-2)-vstar,2);
                
                mingrid(steps) = min(z0(1:end-2));
                maxgrid(steps) = max(z0(1:end-2));
                eval_grid(steps,:) = evals(1:4)';
                eval1_grid(steps) = evals(1);
                eval2_grid(steps) = evals(2);

                %SOL_grid(:,steps) = z0;
                SOL_grid = [SOL_grid;z0];

                % Plot the Real part of the largest 2 eigenvalues
                figure(10)
                plot(1:steps,[eval1_grid(1:steps),eval2_grid(1:steps)],'*-')
                xlabel('$E$','Interpreter','latex','FontSize',18)
                ylabel('Re $\lambda$','Interpreter','latex','FontSize',18)


                set(0,'CurrentFigure',h);

                % plot vbump-v*:
                subplot(3,1,1)
                plot(x,z0(1:end-2)-vstar,'.-')
                xlabel('$x$','Interpreter','latex')
                ylabel('$v-v_*$','Interpreter','latex')

                %plot vbump
                subplot(4,1,2)
                plot(x,z0(1:end-2),'.-')
                xlabel('$x$','Interpreter','latex')
                ylabel('$v$','Interpreter','latex')

                %plot l2 norm of v-v*
                subplot(4,1,3)
                plot(Egrid(1:steps),l2grid(1:steps),'.-')
                xlabel('$E$','Interpreter','latex')
                ylabel('$\| v - v_*\|_{L^2}$','Interpreter','latex')

                %plot min/max of vbump
                subplot(4,1,4)
                plot(Egrid(1:steps),mingrid(1:steps),'b.-')
                hold on
                plot(Egrid(1:steps),maxgrid(1:steps),'r.-')
                xlabel('$E$','Interpreter','latex')
                ylabel('min/max$(v-v_*)$','Interpreter','latex')

                drawnow


                if steps>1
                    sec=z0-zold;
                    sec = sec/norm(sec);%update secant
                end

                ds = min(ds*1.2,arclength_max);

                % march one step in the direction of the previous secant
                zold=z0;
                zinit=z0+ds*sec; % only march after step 1; in step 2 reduce marching since only in direction of parameter
                z0=zinit;
        end % end of newton_control

        % if (abs(mingrid(steps)-Egrid(steps))<0.05 && abs(maxgrid(steps)-Egrid(steps))<0.05) || ()
        %     break
        % end

    end %%%%END CONTINUATION
    %Flip around array after 1st direction
    if pp == par_direc(1)
        Egrid(1:steps) = flipud(Egrid(1:steps));
        cgrid(1:steps) = flipud(cgrid(1:steps));
        l2grid(1:steps) = flipud(l2grid(1:steps));
        SOL_grid(1:steps) = flipud(SOL_grid(1:steps));
        mingrid(1:steps) = flipud(mingrid(1:steps));
        maxgrid(1:steps) = flipud(maxgrid(1:steps));
        eval_grid(1:steps,:) = flipud(eval_grid(1:steps,:));
        eval1_grid(1:steps,:) = flipud(eval1_grid(1:steps,:));
        eval2_grid(1:steps,:) = flipud(eval2_grid(1:steps,:));
    end

end
save('eigen_data.mat','Egrid','cgrid','l2grid','SOL_grid','mingrid','maxgrid','eval_grid','eval1_grid','eval2_grid','N','x','L','J0','J1')

figure(13)
plot(Egrid(1:steps),mingrid(1:steps),'b.-')
hold on
plot(Egrid(1:steps),maxgrid(1:steps),'r.-')
plot(Egrid(1:steps),Egrid(1:steps),'g.-')
plot(Egrid(1:steps),( J0+ sqrt(J0.^2 + 4*(Egrid(1:steps) - J0) ))/2,'y.-')
plot(Egrid(1:steps),( J0- sqrt(J0.^2 + 4*(Egrid(1:steps) - J0) ))/2,'k.-')
