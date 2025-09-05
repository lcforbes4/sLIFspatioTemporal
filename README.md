This is the code repository associated with the paper 'Coherent Dynamics in Networks of Soft-Threshold Integrate-and-Fire Neurons on the Ring.


## Figure 1
- Create Data
   - In `buildFigure1.py` run `two_neuron_delay_visualization_save()` with J_choice = 0. May have to run it multiple times to get a simulation showing the delay nicely.
- Create Figure
  - In `buildFigure1.py` run `build_fig_1_v2()`

## Figure 2
- Create Matlab continuation data
  - From folder `matlab_oscillation_continuation` run `varyingE.m`, `varyingJ.m`, and `varyingDelay.m`
  - Save the 3 resulting .mat files in `current_fig_scripts/figHopf_data/matlab_data`
- Create spiking simulation data
  - Run `Create_Sim_Data_for_hopf_cont_compare.py` 3 times for each of the parameters. Change the inputs for the following cases
    1. param_name_to_vary = 'J0', num_of_points = 5, param_min = -5, param_max = -3.25, and tstop=500
    2. param_name_to_vary = 'E', num_of_points = 10, param_min = 1.05, param_max = 4.1, and tstop=100
    3. param_name_to_vary = 'delay', num_of_points = 10, param_min = .9, param_max = 2, and tstop=100
- Create Figure
  - In `build_Hopf_fig.py` run `build_Hopffig_v2()`

## Figure 3
- Create ramping and pulse input data
  - In `create_data_for_Hopf_bistab_fig.py` run `MF_and_spk_ramp_data_save()` and `spiking_bistable_pulse_data_save()`
- Create figure
  - In `buildFig_osc_bistab.py` run `build_bistab_fig_v2()`
  - Note: The bifurcation diagrams overlaid on the mean field ramping simulations uses the matlab continutation data from Fig 2.

## Figure 4
- Create Matlab continuation data
  - From folder `matlab_bump_continuation` run 
    - `cont_in_E_and_stability.m` with J0 = 5 and name output `lif_data.mat`,
    - `cont_in_J0.m` with E =  and name output `lif_data_J0.mat`,
    - and `cont_in_J1.m` with E = J0 = 5 and name output `lif_data_J1.mat`.
  - Save the 3 resulting .mat files in `current_fig_scripts/spatial_fig_data`
- Create figure 
  - In `build_spatial_fig.py` run `build_spatial_fig_v3()`

## Figure 5
- Create matlab data. From folder `matlab_bump_continuation` 
  - run `tracking_fold_points_cont_in_E_step_in_J0.m`
  - run `cont_in_E_and_stability.m` 4 times, once for each of the values J0 = {4, 5, 5.75, 6.5}, and save output as `lif_cont_in_E_J0_{ }_J1_10.mat`
  - run `cont_in_J0.m` 4 times, once for each of the values E = {1.5, 1, 0.25, -1.5}, and save output as `lif_cont_in_J0_E_{ }_J1_10.mat`
  - Save the 9 resulting .mat files in `current_fig_scripts/spatial_fig_data`
- Create bistable spiking data
  - run `create_spatial_bistability_spk_data.py` 3 times
    1. J0 = 3, E = 0.75, and which_portion = 0, name output `Bistab_QL_spiking_pop_spk_times_J03_J10_E.75.npy`
    2. J0 = 3.1, E = 1.75, and which_portion = 1, name output `Bistab_HQL_spiking_pop_spk_times_J4.2_J10_E.5.npy`
    3. J0 = 4.2, E = 0.5, and which_portion = 1, name output `Bistab_HL_spiking_pop_spk_times_J3.1_J10_E1.75.npy`
  - Save these 3 resulting .npy files in `current_fig_scripts/spatial_fig_data`
- Create figure
  - In `build_spatial_bistab_fig.py` run `build_spatial_fig_v4()`

## Figure 6
- Create data
  - In `create_spatio_temp_data.py` run `run_compare_and_save()` 6 times with the following parameters
    1. name = 'breather', J0 = -60, J1 = 20, E = 10, D = .2, v = 0
    2. name = 'chaoticzigzag', J0 = -5, J1 = -90, E = 10, D = .2, v = 0
    3. name = 'spots', J0 = -5, J1 = -100, E = 10, D = .2, v = 0
    4. name = 'zigzag', J0 = -17, J1 = -60, E = 10, D = .2, v = 0
    5. name = 'lurching', J0 = -16, J1 = -72, E = 10, D = .2, v = 0
    6. name = 'mixedSW1moving', J0 = -25, J1 = -95, E = 10, D = .2, v = 0
    7. name = 'SW', J0 = -2, J1 = -8, E = 2, D = 1, v = 0 
    8. name = 'TW', J0 = -2, J1 = -8, E = 2, D = 1, v = 2
  - This should create a total of 16 .npy files in `current_fig_scripts/spiking_network_patterns/fig`: One mean-field simulation and one spiking network simulation for each set of parameters. 
- Create figure
  - In `spatio_temporal_figure.py` run `build_spatio_tempo_fig()`

## Figure 7
- Create data for the transmission of a spike between 2 neurons with different temporal filters
  - For the exponential function, in `buildFigure1.py` run `two_neuron_delay_visualization_save()` with J_choice = 1.
  - For the alpha function, in `buildFigure1.py` run `two_neuron_delay_visualization_save()` with J_choice = 2.
  - Note: You may have to run multiple times to get a simulation showing the delay nicely.
- Calculate data for the alpha-function Turing-Hopf curve in Mathematica
  - Run the mathematica notebook `alpha_function_TuringHopf_Curve.nb` which should result in two .csv files. Save these in `current_fig_scripts/more_realistic_cases_data`
- Create figure
  - In `more_realistic_syn_response_fig.py` run `build_fig_syn_respose()`

## Figure 8
- Create data
  - From folder `matlab_bump_continuation` run `cont_in_E_and_stability.m` with J0 = 3 and name output `eigen_data.mat`
- Create figure
  - In `plot_stability_cont.py` run `plot_evals()`
