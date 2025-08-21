%% tracking_fold_points_cont_in_E_step_in_J0.m
% This is to continue in J_0 and E with J_1 held constant to get location
% of fold point
clc; clear; close all;

%% Steps:
% - continue in E at some initial condition of J0
% - check if fold present, and grab max/min E value of continuation depending on fold direction
% - step in J_0 and repeat

%% continuation parameters
ds = 0.25;
maxstep = 500;
arclength_max = 0.5;

%second parameter continuation params
num_of_sec_param_steps = 10;
sec_param_step_size = 0.5;

% set domain and initial system parameters
L = 2*pi;
N = 2^12;
x = linspace(0, L,N+1)'; 
x = x(1:end-1);
dx = L/N;
dt = 0.01; % only for time steppers;
J1 = 10;
J0 = 0;
E0 = 1.1; 
par_direc = [1,-1];

%derivative vectors (in Fourier space)
k   = ([[0:N/2] [-N/2+1: -1]])';

%newton parameters
tol = 1e-6;
niter = 100;
maxiter = 20;
nminstep = 1e-14;
nitermax =10;
newtonflag=0;


sx = sin(x);


% firing rate function
f = @(v) (v-1).*heaviside(v-1);
f1 = @(v) heaviside(v-1);

%% set up figure:
h=figure(4); % to plot E continutation
h2 = figure(44); % to plot fold point

%%

%initialize arrays
J0_orig = J0;
Efoldgrid_top = zeros(num_of_sec_param_steps,1);
Eturinggrid_top = Efoldgrid_top;
Efoldgrid_bottom = ones(num_of_sec_param_steps,1);
Eturinggrid_bottom = Efoldgrid_top;
J0foldgrid = Efoldgrid_top;

for second_param_steps = 0:num_of_sec_param_steps-1
    % Recompute
    J0 = J0_orig + sec_param_step_size*second_param_steps;
    E0 = -1.444*J0+6.582; %Sometimes need to change ICs to get initial convergence, this chooses E to be a little above the Turing value

    %kernel
    J = (J0 + J1*cos(x))/ 2/pi;
    Jhat = fft(J);   
    
    %% Run Euler time-stepper    
    totaltime = 1;
    plotevery = 5;
    
    % Initial Condition Choices:
    equil_choice = 0; % 0 for v+, 1 for v-, 2 for v_0=E
    pertstrength = 10;%1e-1;
    spatialdependence = 1; % 0 for spatially homogeneous perturbation, 1 for perturb*cos

    % run time stepper to get good initial guess:
    
    %switch to v- if after co-dim 2 point
    if J0>4.5 %codim 2 pt occurs at about 4.5 for J1=10
        pertstrength = .1;
        equil_choice=1;
        E0 = 0;
        par_direc=[-1,1];
    end
    v0 = meanfield_v2(J0,J1,E0,N, dt, totaltime, equil_choice, pertstrength, spatialdependence, plotevery);
       
    %% Initial continuation 
    u=[v0;0]; % append parameter line
    F0 =@(uu) lif_u(uu, Jhat, E0,f,k,sx,N); %Initial root find?
    
    % Set linear solver tolerance to be about 10 times smaller than the Newton
    gmrestol=tol*1e-2;
    gmaxit = 100;
    gminner = 10;
    rhs = F0(u);
    residual = norm(rhs);
    
    % % plot
    % figure(2)
    % plot(rhs)
    % title('F0([v0;0]) RHS')
    
    % Newton loop using GMRES (with no preconditioner) to solve linear equation
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
            % % Plot solution as its converging
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
    c0 = u(end);
    
    % % Plot solution profile at initial parameters
    % figure(3)
    % plot(x,u(1:end-1))
    % title('result of convergence at initial parameters')
    
    
    
    %% Secant continuation in parameter E
    
    % u is the solution of initial convergence
    vplus = (J0+sqrt(J0^2+4*(E0-J0)))/2;
    vminus = (J0+sqrt(J0^2+4*(E0-J0)))/2;

    Egrid = zeros(maxstep,1);   Egrid(1) = E0; 
    cgrid = Egrid;              cgrid(1) = c0;

    %to track L2 distance from each of the equilibria:
    l2grid_E = Egrid;             l2grid_E(1) = sqrt(dx)*norm(u(1:end-2)-E0,2);
    l2grid_vminus = Egrid;      l2grid_vminus(1) = sqrt(dx)*norm(u(1:end-2)-vminus,2);
    l2grid_vplus = Egrid;      l2grid_vplus(1) = sqrt(dx)*norm(u(1:end-2)-vplus,2);

    SOL_grid = [u;E0];
    mingrid = Egrid;            mingrid(1) = min(u(1:end-1));
    maxgrid = Egrid;            maxgrid(1) = max(u(1:end-1));
    
    steps = 1;
    
    for pp = par_direc
        flip = 0;
    
        %initialize secant, stepping in par_direc
        sec = [zeros(N+1,1);pp];
        zold = [u;E0]; % The initial solution
        z0 = zold+0.01*sec; % step it in one direction of the parameter
        zinit = z0; % the stepped version as initial condition of continuation
        ii=1;
    
        %%%%%BEGIN continuation %%%%%
        while (steps < maxstep) &&  (sqrt(dx)*norm(z0(1:end-2)-( (J0+sqrt(J0^2+4*(z0(end)-J0)))/2),2)>0.3) && (sqrt(dx)*norm(z0(1:end-2)-( (J0-sqrt(J0^2+4*(z0(end)-J0)))/2),2)>0.3) && (sqrt(dx)*norm(z0(1:end-2)-z0(end),2)>0.3)
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
                %
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
                    steps = steps+1;EE = z0(end);
                    Egrid(steps) = EE; 
                    cgrid(steps) = z0(end-1);

                    vplus = (J0+sqrt(J0^2+4*(EE-J0)))/2;
                    vminus = (J0-sqrt(J0^2+4*(EE-J0)))/2;

                    l2grid_E(steps) = sqrt(dx)*norm(z0(1:end-2)-EE,2);
                    l2grid_vplus(steps) = sqrt(dx)*norm(z0(1:end-2)-vplus,2);
                    l2grid_vminus(steps) = sqrt(dx)*norm(z0(1:end-2)-vminus,2);
                    
                    mingrid(steps) = min(z0(1:end-2));
                    maxgrid(steps) = max(z0(1:end-2));
                    SOL_grid = [SOL_grid;z0];
    
                    % Plot Continuation in E
                    set(0,'CurrentFigure',h);
                    %subplot(4,4,second_param_steps+1)
                    plot(Egrid(1:steps),Egrid(1:steps),'g.-')
                    plot(Egrid(1:steps),( J0+ sqrt(J0.^2 + 4*(Egrid(1:steps) - J0) ))/2,'y.-')
                    plot(Egrid(1:steps),( J0- sqrt(J0.^2 + 4*(Egrid(1:steps) - J0) ))/2,'k.-')
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
    
                    ds = min(ds*1.4,2);
    
                    % march one step in the direction of the previous secant
                    zold=z0;
                    zinit=z0+ds*sec; % only march after step 1; in step 2 reduce marching since only in direction of parameter
                    z0=zinit;
            end % end of newton_control
    
        end %%%%END CONTINUATION

        %Flip around arrays after 1st direction:
        if pp == par_direc(1)
            Egrid(1:steps) = flipud(Egrid(1:steps));
            cgrid(1:steps) = flipud(cgrid(1:steps));
            l2grid_E(1:steps) = flipud(l2grid_E(1:steps));
            l2grid_vplus(1:steps) = flipud(l2grid_vplus(1:steps));
            l2grid_vminus(1:steps) = flipud(l2grid_vminus(1:steps));
            SOL_grid(1:steps) = flipud(SOL_grid(1:steps));
            mingrid(1:steps) = flipud(mingrid(1:steps));
            maxgrid(1:steps) = flipud(maxgrid(1:steps));
        end
    
    end

    % Save J0 and E locations of the Turing and non-smooth Turing
    J0foldgrid(second_param_steps+1) = J0;
    Eturinggrid_top(second_param_steps+1) = Egrid(1); %Turing
    Eturinggrid_bottom(second_param_steps+1) = Egrid(steps); %non-smooth Turing
    
    % plot turing locations in J0-E plane:
    set(0,'CurrentFigure',h2);
    plot(J0foldgrid(1:second_param_steps+1),Eturinggrid_top(1:second_param_steps+1),'b.-')
    hold on
    plot(J0foldgrid(1:second_param_steps+1),Eturinggrid_bottom(1:second_param_steps+1),'b.-')

    % Save and plot fold points if exist:
    upper_fld_idx = detect_upper_fold(Egrid(1:steps));
    lower_fld_idx = detect_lower_fold(Egrid(1:steps));
    if ~isnan(upper_fld_idx)
        Efoldgrid_top(second_param_steps+1) = Egrid(upper_fld_idx);
        plot(J0foldgrid(1:second_param_steps+1),Efoldgrid_top(1:second_param_steps+1),'r.-')
    end
    if ~isnan(lower_fld_idx)
        Efoldgrid_bottom(second_param_steps+1) = Egrid(lower_fld_idx);%min(Egrid(1:steps));
        plot(J0foldgrid(1:second_param_steps+1),Efoldgrid_bottom(1:second_param_steps+1),'r.-')
    end

    % plot saddle bif:
    if J0>=2
        plot(J0foldgrid(1:second_param_steps+1),1 - ((J0foldgrid(1:second_param_steps+1)-2)/2).^2,'g.-')
    end
    drawnow


end

save('fold_cont_data.mat','J0foldgrid','Eturinggrid_top','Efoldgrid_top','Eturinggrid_bottom','Efoldgrid_bottom','N','x','L','J1','num_of_sec_param_steps')