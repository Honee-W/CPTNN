# CPTNN unofficial implementation
## original paper: "CPTNN: CROSS-PARALLEL TRANSFORMER NEURAL NETWORK FOR TIME-DOMAIN SPEECH ENHANCEMENT"

### single-channel time domain speech enhancement neural network
----

### How to use:
  step1: add cptnn_new.py, TRANSFORMER.py, process_for_cptnn.py to your model directory. 
  
  step2: import and ready to go.
  
  
### configuration:
    current params: 1.1M
    
    frame_len, hop_size: transform wavform to segments
    
    feat_dim, hidden_size, num_heads, cptm_layers: tune your hyperparameters based on your task
