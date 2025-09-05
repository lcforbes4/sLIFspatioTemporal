%% my_file.m
%varies J
clc;clear;

%%

%initial conditions on where to start equil branch
J = -4;
E = 3;
tau = 2;

%equil branch
branch1_min = -5;
branch1_max = -1;
branch1_step = 0.1;

% hopf curve branch
branch2_min = -5;
branch2_max = -1;
branch2_step = 0.1;

% periodic sol branch
branch4_min = -5;
branch4_max = -3;
branch4_step = 0.00005;
branch4_num_of_points = 600;




%% Right-hand side
% A function defining the right-hand side $f$:
% 
%   function y=sys_rhs(xx,par)
% 
% This function has two arguments, |x| $\in R^{2}$, which contains 
% the state variable(s) at the present and in the past,
% |x| $=[x(t), x(t-D)]$,
% |par| $\in R^{3}$ which contains the parameters, 
% |par| $=[E, J0]$.

% f = -v -v f(v) + J f(v-D) + E
my_sys_rhs=@(v,par)[-v(1)-v(1)*max(0,v(1)-1)+par(1)*max(0,v(2)-1)+par(2)];

%% Delays
% -The delay $D$ is considered to be part of the
% parameters 
% -DDE-Biftool calls their delays taus
% -Note: the order of the parameters as 
% |par| $=[J,E,D]$.
%

my_tau=@()[3];

% Saving the index of our bifurction parameters in par
ind_J=1;  % used later for continuation
ind_E=2; % used later for continuation

%% Definition of structure |funcs|
% 

funcs=set_funcs(...
    'sys_rhs',my_sys_rhs,...
    'sys_tau',my_tau) %#ok<NOPTS>


%% Initial guess for steady state
% It is clear that the neuron DDE has a steady state solution
% $v=(J+-sqrt(J^2+4(E-J)))/2$ for all values of the parameters. 
% 
% We define a first
% steady state solution using the initial parameter values
%
% Remember that we chose 
% |par| $=[J,E,\tau]$.


v0=(J+sqrt(J^2+4*(E-J)))/2; %used + vers

stst.kind='stst';
stst.parameter = [J E tau]
stst.x = [v0]


%% Linear stability of initial equilibrium
% We get default point method parameters and correct the point, which,
% being already a correct solution, remains unchanged. Computing and
% plotting stability of the corrected point

flag_newhheur=1; % flag_newhheur=1 is the default choice if this argument is omitted
method=df_mthod(funcs,'stst',flag_newhheur);
method.stability.minimal_real_part=-2
[stst,success]=p_correc(funcs,stst,[],[],method.point)
% compute its stability:
stst.stability=p_stabil(funcs,stst,method.stability)
figure(1); clf;
subplot(3,4,[1,2,5,6])
p_splot(stst); % plot its stability:
title('Eigenvalues of Initial Equilibrium')

%% Initialize branch of trivial equilibria
% We will use this point as a first point to compute a branch of steady
% state solutions. First, we obtain an empty branch with free parameter
% J, limited by bounds and step size between points set above
subplot(3,4,[3,4]);
% get an empty branch with J as a free parameter:
branch1=df_brnch(funcs,ind_J,'stst')
branch1.parameter
branch1.parameter.min_bound
% set bounds for continuation parameter
branch1.parameter.min_bound(1,:)=[ind_J branch1_min];
branch1.parameter.max_bound(1,:)=[ind_J branch1_max];
branch1.parameter.max_step(1,:)=[ind_J branch1_step];
% use stst as a first branch point:
branch1.point=stst;

%%  Extend and continue branch of trivial equilibria
% To obtain a second starting point we change  parameter value $J$
% slightly and correct again.

stst.parameter(ind_J)=stst.parameter(ind_J)+0.05;
[stst,success]=p_correc(funcs,stst,[],[],method.point)
% use as a second branch point:
branch1.point(2)=stst;
branch1.method.continuation.plot_progress=0;

% continue in one direction:
[branch1,s,f,r]=br_contn(funcs,branch1,100)
% turn the branch around:
branch1=br_rvers(branch1);
% continue in the other direction:
[branch1,s,f,r]=br_contn(funcs,branch1,100)

title('Equilibrium values for varied parameter')
xlabel('J');ylabel('v');

%% Stability of branch of equilibria
% 
mrp=-2; %minmal real part of lambda
Mrp=2; %maximal real part of lambda
minJ=branch1_min;
maxJ=branch1_max;


branch1.method.stability.minimal_real_part=mrp;
branch1=br_stabl(funcs,branch1,0,0);

% obtain suitable scalar measures to plot stability along branch:
[xm,ym]=df_measr(1,branch1)

subplot(3,4,[7,8]);
br_plot(branch1,xm,ym,'b'); % plot stability along branch:
ym.subfield='l0';%?? idk what this is
br_plot(branch1,xm,ym,'c');
plot([minJ maxJ],[0 0],'-.'); %manually plot equil line
axis([minJ maxJ mrp Mrp]);
title('Stability along branch')
xlabel('J');ylabel('\Re(\lambda)');

% plot stability versus point number:
subplot(3,4,[9,10,11,12]);
br_plot(branch1,[],ym,'b');
br_plot(branch1,[],ym,'b.');
plot([0 30],[0 0],'-.'); %plot Re(lam)=0
title('Stability vs Point Number')
xlabel('point number along branch');ylabel('\Re(\lambda)');

%% Locating the first Hopf point
% Where eigenvalue curves in the stability plot
% cross the zero line, bifurcations occur. This is
% most easily obtained by plotting the stability versus the point numbers
% along the branch.

%We select the last point with positive eigenvalues and
% turn it into an (approximate) Hopf bifurcation point:
ind_hopf=find(arrayfun(@(x)real(x.stability.l0(1))< 0,branch1.point),1,'last');
hopf=p_tohopf(funcs,branch1.point(ind_hopf));

%We correct the Hopf
% point using appropriate method parameters and one free parameter J:
method=df_mthod(funcs,'hopf',flag_newhheur); % get hopf calculation method parameters:
method.stability.minimal_real_part=-1;
[hopf,success]=p_correc(funcs,hopf,ind_J,[],method.point) % correct hopf

%We then copy the corrected point to keep it for later use.:
first_hopf=hopf;                    % store hopf point in other variable for later use

hopf.stability=p_stabil(funcs,hopf,method.stability); % compute stability of hopf point
figure(2); clf;
subplot(2,3,[1,4])
p_splot(hopf);                     % plot stability of hopf point
title('Eigenvalues at Hopf-Bifurcation')

%% Initialize and continue first Hopf bifurcation
% In order to follow a branch of Hopf bifurcations in the two parameter
% space $(J0,E)$ we again need two starting points. Hence we use
% the Hopf point already found and one perturbed in $E$ and corrected
% in $J0$, to start on a branch of Hopf bifurcations. 
% We continue the branch on both sides by
% an intermediate order reversal and a second call to |br_contn|.
branch2=df_brnch(funcs,[ind_J,ind_E],'hopf'); % use hopf point as first point of hopf branch:
branch2.parameter.min_bound(1,:)=[ind_J branch2_min];
branch2.parameter.max_bound(1:2,:)=[[ind_J branch2_max]' [ind_E 10]']';
branch2.parameter.max_step(1:2,:)=[[ind_J branch2_step]' [ind_E 0.1]']';
branch2.point=hopf;

hopf.parameter(ind_E)=hopf.parameter(ind_E)+0.1; % perturb hopf point
[hopf,success]=p_correc(funcs,hopf,ind_J,[],method.point); % correct hopf point, recompute stability
branch2.point(2)=hopf; % use as second point of hopf branch:

subplot(2,3,[2,3])
[branch2,s,f,r]=br_contn(funcs,branch2,40);            % continue with plotting hopf branch:
branch2=br_rvers(branch2);                             % reverse Hopf branch
[branch2,s,f,r]=br_contn(funcs,branch2,30);            % continue in other direction
title('Hopf Curve in J-E Plane')
xlabel('J');ylabel('E');

%% Hopf continuation and detecton of Takens-Bogdanov point
% As we did not change continuation method parameters, predictions and
% corrections will be plotted during continuation. The final result is
% shown as figure. We compute and plot stability along the branch.
branch2=br_stabl(funcs,branch2,0,0);

subplot(2,3,[5,6])
[xm,ym]=df_measr(1,branch2); % plot stability versus point number:
ym.subfield='l0';
br_plot(branch2,[],ym,'c');
ym.subfield='l1';
br_plot(branch2,[],ym,'b');
title('Stability of Eigenvalues along Hopf Curve')
xlabel('point number along branch');ylabel('\Re(\lambda)');

%% Constructing an initial small-amplitude orbit near a Hopf bifurcation
% We use the first Hopf point we computed (|first_hopf|) to construct a
% small-amplitude (|1e-2|) periodic solution on an equidistant mesh of 18
% intervals with piecewise polynomial degree 3. The steplength condition
% (returned by |p_topsol|) ensures the branch switch from the Hopf to the
% periodic solution as it avoids convergence of the amplitude to zero
% during corrections. Due to the presence of the steplength condition we
% also need to free one parameter, here J0
intervals=20;
degree=3;
[psol,stepcond]=p_topsol(funcs,first_hopf,1e-2,degree,intervals);
% correct periodic solution guess:
method=df_mthod(funcs,'psol');
[psol,success]=p_correc(funcs,psol,ind_J,stepcond,method.point);

%% Construction and continuation of branch
% The result, along with a degenerate periodic solution with amplitude zero
% is used to start on the emanating branch of periodic solutions, see
% figure below. We avoid adaptive mesh selection and save
% memory by clearing the mesh field. An equidistant mesh is then
% automatically used which is kept fixed during continuation. Simple
% clearing of the mesh field is only  possible if it is already
% equidistant. This is the case here as |p_topsol| returns a solution on an
% equidistant mesh.
branch4=df_brnch(funcs,ind_J,'psol'); % empty branch:
branch4.parameter.min_bound(1,:)=[ind_J branch4_min];
branch4.parameter.max_bound(1,:)=[ind_J branch4_max];
branch4.parameter.max_step(1,:)=[ind_J branch4_step];

% make degenerate periodic solution with amplitude zero at hopf point:
deg_psol=p_topsol(funcs,first_hopf,0,degree,intervals);
% use deg_psol and psol as first two points on branch:
%deg_psol.mesh=[];
branch4.point=deg_psol;
%psol.mesh=[];
branch4.point(2)=psol;

figure(3);clf;
%subplot(3,4,[3,4]);%keep # of trials low to see concavity
[branch4,s,f,r]=br_contn(funcs,branch4,branch4_num_of_points); % compute periodic solutions branch
xlabel('J');ylabel('amplitude');

%% Combine Amplitude Plot with equilibrium plot
figure(4);clf;

len=length(branch4.point);

%grab J and amps values of each point
for i=1:len
    point(i) = branch4.point(i);
    J_temp(i)=point(i).parameter(1);
    max_temp(i)=max(point(i).profile(:));
    min_temp(i)=min(point(i).profile(:));
    equil_val_temp(i)=(J_temp(i)+sqrt(J_temp(i)^2+4*(E-J_temp(i))))/2;
end

hold on
plot(J_temp,equil_val_temp)
plot(J_temp,max_temp)
plot(J_temp,min_temp)







%% Stability of periodic orbits
% We compute and plot the stability (approx Floquet multipliers) just before and
% after the turning point. The second spectrum is clearly unstable but no
% accurate trivial Floquet multiplier is present at 1.
% 
% 
%% Plot a grid of Floquet Multipliers for some of the points in branch 4
% figure(5); clf;
% starting_index = 18;
% len = 4;
% for i=1:len*len
%     psol=branch4.point(starting_index+i);
%     psol.stability=p_stabil(funcs,psol,method.stability);
%     subplot(len,len,i);
%     p_splot(psol);
%     title(['Point number ',num2str(starting_index+i)])
% end

% Compute the stability for all of branch4
for i=1:len
    branch4.point(i).stability = p_stabil(funcs,branch4.point(i),method.stability);
end

save("branch4_Jsave.mat","branch4","-v7")


%% Find Fold of Periodic Points
% addpath '/Users/lcforbes/Documents/MATLAB/dde_biftool_v3.1.1/ddebiftool_extra_psol'
% addpath '/Users/lcforbes/Documents/MATLAB/dde_biftool_v3.1.1/ddebiftool_utilities'
% 
% 
% my_sys_rhs=@(v,par)[-v(1,1,:)-v(1,1,:).*max(0,v(1,1,:)-1)+par(1)*max(0,v(1,2,:)-1)+par(2)];
% 
% vfuncs=set_funcs(...
%     'sys_rhs',my_sys_rhs,...
%     'sys_tau',@()[3],...
%     'x_vectorized',true);
% 
% [dummy,indmax]=max(arrayfun(@(x)x.parameter(ind_J),branch4.point));
% %ind = 409;
% %nunst_per=GetStability(branch4,'exclude_trivial',true);
% 
% disp('Find and continue fold of periodic orbits in J and E');
% ind_fold=indmax%find(nunst_per==0,1,'first')-1;
% branch4.parameter.max_step=[1,0.5]; % remove step size restriction
% [pfuncs,pbranch,suc]=SetupPOfold(vfuncs,branch4,ind_fold,'contpar',[ind_J,ind_E],'dir',ind_E,'step',-1e-4,'print_residual_info',1, ...
%     'max_step',[ind_J,0.001; ind_E,0.001]);
% 
% if suc
%     disp('POFold initialization finished');
% else
%     warning('POFold initialization failed');
% end
% 
% %% Continue the fold in J and E toward decreasing E
% figure(1);
% pbranch=br_contn(pfuncs,pbranch,60);