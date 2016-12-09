#!/bin/bash
#Experiments for December 9th
#first experiment just run model for 30 timesteps
python observed_image_seq2output_image_seq.py "thirty_tsteps_lr_1e-3/" "summary_thirty_tsteps_lr_1e-3" "True" "30" "1e-3" "True"
#same as first but with smaller learning rate
python observed_image_seq2output_image_seq.py "thirty_tsteps_lr_1e-4/" "summary_thirty_tsteps_lr_1e-4" "True" "30" "1e-4" "True"
#now change the number of tsteps to 90
python observed_image_seq2output_image_seq.py "ninety_tsteps_lr_1e-3/" "summary_ninety_tsteps_lr_1e-3" "True" "90" "1e-3" "True"
#now do the full sequence length
python observed_image_seq2output_image_seq.py "all_tsteps_lr_1e-3/" "summary_alltsteps_lr_1e-3" "False" "30" "1e-3" "True"
#now do the full sequence but with the observed image fed into the evironment network rather than the previous output
python observed_image_seq2output_image_seq.py "all__tsteps_lr_1e-3_obs/" "summary_alltsteps_lr_1e-3_obs" "False" "30" "1e-3" "False"
