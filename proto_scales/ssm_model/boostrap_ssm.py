import proto_scales.ssm_model.scales_ssm_z2tasAndpr_hysteresis as scales_ssm_z2pr_diagonal
import proto_scales.ssm_model.scales_ssm_tas_pr_hyst_correlated as scales_ssm_z2pr_correlated

zdim = 64
rnn_hidden=256
use_linear_model = True
emission_uses_u =True
alpha_max = 0.002
resevoir_dim = 4

scaler_path ="/home/kainverena/PythonProjects/outputs_ssm_scales/scales_20260514_171221/"
model_filename = scaler_path+"checkpoints/model_epoch0500.pt" 

